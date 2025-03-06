import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
import opacus

from Optimizers.Adam_optimizer.AdamBC import *
from privacy_engines.Dice_privacy_engine import Dice_PrivacyEngine
from privacy_engines.KFprivacy_engine import KF_PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
class Training:


    @staticmethod 
    def training_loop(model_type,num_epochs, train_loader, model, criterion, optimizer, device, verbose=False, use_closure=False):
        all_losses = []
        all_accuracies = []
        if model_type.lower() == 'classification':
            all_losses,all_accuracies = Training.classification_training_loop(num_epochs,train_loader,model,criterion,optimizer,device,verbose=verbose,use_closure=use_closure)

        elif model_type.lower() == 'vae':
            all_losses = Training.VAE_training_loop(num_epochs,train_loader,model,criterion,optimizer,device,verbose=verbose,use_closure=use_closure)
        else:
            raise ValueError("Unsupported model type")
        return all_losses, all_accuracies
    @staticmethod 
    def classification_training_loop(num_epochs, train_loader, model, criterion, optimizer, device, verbose=False, use_closure=False):
        all_losses = []
        all_accuracies = []
        model.to(device)
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_accuracies = []
            for x, y in tqdm(train_loader, desc=f'{epoch+1}/{num_epochs}'):
                x, y = x.to(device), y.to(device)

                if use_closure:
                    def closure():
                        out = model(x)  # Forward pass
                        loss = criterion(out, y)  # Compute the loss
                        loss.backward()  # Backward pass
                        return loss, out
                    loss, out = optimizer.step(closure)
                    optimizer.zero_grad()

                else:
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_losses.append(loss.item())
                # Calculate accuracy
                _, predicted = torch.max(out.data, 1)
                accuracy = (predicted == y).sum().item() / y.size(0)
                epoch_accuracies.append(accuracy)

            all_losses.append(epoch_losses)
            all_accuracies.append(epoch_accuracies)
            if verbose:
                print(f"Epoch {epoch + 1}, loss  = {np.sum(epoch_losses) / len(epoch_losses) }, accuracy = {np.sum(all_accuracies) / len(all_accuracies)}")

        return all_losses, all_accuracies
    
    @staticmethod
    def VAE_training_loop(num_epochs, train_loader, model, criterion, optimizer, device, verbose=False, use_closure=False):
        all_losses = []
        model.to(device)
        for epoch in range(num_epochs):
            epoch_losses = []
            for x, _ in tqdm(train_loader, desc=f'{epoch+1}/{num_epochs}'):
                x = x.to(device)

                if use_closure:
                    def closure():
                        recon_x, mu, logvar = model(x)  # Forward pass
                        loss = criterion(recon_x, x, mu, logvar)  # Compute the loss
                        loss.backward()  # Backward pass
                        return loss, recon_x
                    loss, recon_x = optimizer.step(closure)
                    optimizer.zero_grad()

                else:
                    recon_x, mu, logvar = model(x)
                    loss = criterion(recon_x, x, mu, logvar)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_losses.append(loss.item())

            all_losses.append(epoch_losses)
            if verbose:
                print(f"Epoch {epoch + 1},  loss = {np.sum(epoch_losses) / len(epoch_losses)}")

        return all_losses
    

    @staticmethod
    def train(optimizer_type,
              model,
              train_loader, 
              learning_rate, 
              sample_rate, 
              criterion, 
              num_epochs, 
              target_epsilon, 
              clip_bound, 
              delta, 
              device, 
              verbose=False, 
              **kwargs):
        optimizer_type = optimizer_type.upper()
        if optimizer_type == 'SGD':
            return Training.private_train_epsilon(model, train_loader,  learning_rate, sample_rate, criterion, num_epochs, target_epsilon, clip_bound, delta, device, verbose=verbose, **kwargs)
        elif optimizer_type == 'DICE':
            return Training.DP_Dice_train_epsilon(model, train_loader,  learning_rate, sample_rate, criterion, num_epochs, target_epsilon, clip_bound, delta, device, verbose=verbose, **kwargs)
        elif optimizer_type == 'ADAM':
            return Training.DP_AdamBC_train_epsilon(model, train_loader,  learning_rate, sample_rate, criterion, num_epochs, target_epsilon, clip_bound, delta, device, verbose=verbose, **kwargs)
        elif optimizer_type == 'KF':
            return Training.DP_KF_train_epsilon(model, train_loader,  learning_rate, sample_rate, criterion, num_epochs, target_epsilon, clip_bound, delta, device, verbose=verbose, **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def private_train_epsilon(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        learning_rate,
        sample_rate,
        criterion,
        num_epochs: int,
        target_epsilon: float,
        clip_bound: float,
        delta: float,
        device,
        normalize_clipping= False,
        random_seed=474237,
        verbose=False,
        **kwargs):
        # this is a known warning that can be safely ignored.
        warnings.filterwarnings(
            "ignore", message="Using a non-full backward hook when the forward contains multiple autograd Nodes"
        )
        # we would use a secure RNG in production but here we can ignore it.
        warnings.filterwarnings("ignore", message="Secure RNG turned off.")
        # this accounting warning is not optimal but fine to ignore for this example
        warnings.filterwarnings("ignore", message="Optimal order is the largest alpha.")


        privacy_engine = opacus.PrivacyEngine(
            accountant="prv",
            secure_mode=False,  # Should be set to True for production use
        )

        optimizer = optim.SGD(model.parameters(), learning_rate)
        rng = torch.Generator(device=device)
        rng.manual_seed(int(random_seed))

        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            target_delta=delta,
            epochs=num_epochs,
            max_grad_norm=clip_bound,
            noise_generator=rng,
            loss_reduction="mean",
            normalize_clipping = normalize_clipping,
        )
        all_losses,all_accuracies = Training.training_loop(kwargs['model_type'],num_epochs,train_loader,model,criterion,optimizer,device,verbose=verbose)
        epsilon = privacy_engine.get_epsilon(delta)
        return epsilon, all_losses,all_accuracies
    
        
    @staticmethod
    def DP_Dice_train_epsilon(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        learning_rate,
        sample_rate,
        criterion,
        num_epochs: int,
        target_epsilon: float,
        clip_bound: float,
        delta: float,
        device,
        normalize_clipping= False,
        random_seed=474237,
        verbose=False,
        **kwargs):
        # this is a known warning that can be safely ignored.
        warnings.filterwarnings(
            "ignore", message="Using a non-full backward hook when the forward contains multiple autograd Nodes"
        )
        # we would use a secure RNG in production but here we can ignore it.
        warnings.filterwarnings("ignore", message="Secure RNG turned off.")
        # this accounting warning is not optimal but fine to ignore for this example
        warnings.filterwarnings("ignore", message="Optimal order is the largest alpha.")


        # privacy engine
        # privacy_engine = opacus.PrivacyEngine(
        optimizer = optim.SGD(model.parameters(), learning_rate)
        privacy_engine = Dice_PrivacyEngine(
            accountant="prv",
            # accountant='rdp',
            secure_mode=False,  # Should be set to True for production use
        )

        rng = torch.Generator(device=device)
        rng.manual_seed(int(random_seed))

        # making the model and optimizer private
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            target_delta = delta,
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            epochs=num_epochs,
            max_grad_norm=clip_bound,
            noise_generator=rng,
            loss_reduction="mean",
            normalize_clipping = normalize_clipping,
            error_max_grad_norm = kwargs['error_max_grad_norm'] ,
            **kwargs
        )



        # Training Loop
        all_losses,all_accuracies = Training.training_loop(kwargs['model_type'],num_epochs,train_loader,model,criterion,optimizer,device,verbose=verbose)
        epsilon = privacy_engine.get_epsilon(delta)
        return epsilon, all_losses,all_accuracies
    

    @staticmethod
    def DP_AdamBC_train_epsilon(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        learning_rate,
        sample_rate,
        criterion,
        num_epochs: int,
        target_epsilon: float,
        clip_bound: float,
        delta: float,
        device,
        normalize_clipping= False,
        random_seed=474237,
        verbose=False,
        **kwargs):
        # this is a known warning that can be safely ignored.
        warnings.filterwarnings(
            "ignore", message="Using a non-full backward hook when the forward contains multiple autograd Nodes"
        )
        # we would use a secure RNG in production but here we can ignore it.
        warnings.filterwarnings("ignore", message="Secure RNG turned off.")
        # this accounting warning is not optimal but fine to ignore for this example
        warnings.filterwarnings("ignore", message="Optimal order is the largest alpha.")



        
        
        # privacy engine


        privacy_engine = opacus.PrivacyEngine(
            accountant="prv",
            # accountant='rdp',
            secure_mode=False,  # Should be set to True for production use
        )
        noise_multiplier = get_noise_multiplier(
                    target_epsilon=target_epsilon,
                    target_delta=delta,
                    sample_rate=sample_rate,
                    epochs=num_epochs,
                    accountant=privacy_engine.accountant.mechanism(),
                    # Add any additional keyword arguments if needed
                )

        # print(noise_multiplier)
        optimizer=AdamCorr(
            model.parameters(), lr=learning_rate,
            dp_batch_size=int(len(train_loader.dataset) * sample_rate),
            dp_l2_norm_clip=clip_bound,dp_noise_multiplier=noise_multiplier
        )
        rng = torch.Generator(device=device)
        rng.manual_seed(int(random_seed))

        # making the model and optimizer private
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            target_delta = delta,
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            epochs=num_epochs,
            max_grad_norm=clip_bound,
            noise_generator=rng,
            loss_reduction="mean",
            normalize_clipping = normalize_clipping,
        )


        # Training Loop
        all_losses,all_accuracies = Training.training_loop(kwargs['model_type'],num_epochs,train_loader,model,criterion,optimizer,device,verbose=verbose)
        epsilon = privacy_engine.get_epsilon(delta)
        return epsilon, all_losses,all_accuracies
    
    @staticmethod
    def DP_KF_train_epsilon(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        learning_rate,
        sample_rate,
        criterion,
        num_epochs: int,
        target_epsilon: float,
        clip_bound: float,
        delta: float,
        device,
        normalize_clipping= False,
        random_seed=474237,
        
        verbose=False,
        **kwargs):
        # this is a known warning that can be safely ignored.
        warnings.filterwarnings(
            "ignore", message="Using a non-full backward hook when the forward contains multiple autograd Nodes"
        )
        # we would use a secure RNG in production but here we can ignore it.
        warnings.filterwarnings("ignore", message="Secure RNG turned off.")
        # this accounting warning is not optimal but fine to ignore for this example
        warnings.filterwarnings("ignore", message="Optimal order is the largest alpha.")



        
        
        # privacy engine


        privacy_engine = KF_PrivacyEngine(
            accountant="prv",
            # accountant='rdp',
            secure_mode=False,  # Should be set to True for production use
        )


        # print(noise_multiplier)
        optimizer = optim.SGD(model.parameters(), learning_rate)
        rng = torch.Generator(device=device)
        rng.manual_seed(int(random_seed))

        # making the model and optimizer private
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            target_delta = delta,
            epochs=num_epochs,
            max_grad_norm=clip_bound,
            noise_generator=rng,
            loss_reduction="mean",
            kalman=True, # need this argument
            kappa=0.7, # optional
            gamma=0.5,# optional
            normalize_clipping = normalize_clipping,

        )


        all_losses,all_accuracies = Training.training_loop(kwargs['model_type'],num_epochs,train_loader,model,criterion,optimizer,device,verbose=verbose,use_closure=True)
        epsilon = privacy_engine.get_epsilon(delta)
        return epsilon, all_losses,all_accuracies