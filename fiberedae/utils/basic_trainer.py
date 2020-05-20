import torch
import numpy
from . import persistence as vpers
from collections import defaultdict

class TrainerHooks(object):
    """docstring for Hook"""
    def __init__(self, callables=None):
        super(TrainerHooks, self).__init__()
        if callables is not None:
            self.callables = callables
        else :
            self.callables = []
    
    def add(self, callables):
        self.callables.extend(callables)

    def run(self, model, trainer):
        for cal in self.callables:
            cal(model, trainer)

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

class SaveBest(object):
    """WIP"""
    def __init__(self, cliff, improvement, filename, overwrite, loss_name):
        super(SaveBest, self).__init__()
        self.cliff = cliff
        self.improvement = improvement
        self.filename = filename
        self.overwrite = overwrite
        self.loss_name = loss_name
        self.last_loss = None

    def run(self, model, trainer):
        if trainer.meta["current_epoch_id"] < self.cliff:
            return

        if trainer.meta["train_loss_history"][self.loss_name] - self.improvement < self.last_loss:
            fn = self.filename
            if not self.overwrite:
                fn = "%s-epoch=%s" % (fn, trainer.meta["current_epoch_id"])
            PERS.save_model(
                model,
                trainer.optimizer,
                history,
                meta_data = {},
                condition_encoding = dataset["label_encoding"],
                model_creation_args = model_args,
                filename=model_save_fn
            )
            self.last_loss = trainer.meta["train_loss_history"][self.loss_name]

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, l_contraction, ignore_sample_zeros=False):
        super(Trainer, self).__init__()
        self.hooks = {
            "starts": TrainerHooks(),
            "stops": TrainerHooks(),
            "epoch_starts": TrainerHooks(),
            "epoch_stops": TrainerHooks(),
            "train_pass_starts": TrainerHooks(),
            "train_pass_stops": TrainerHooks(),
            "test_pass_starts": TrainerHooks(),
            "test_pass_stops": TrainerHooks(),
        }

        self.meta = {
            "batch_loss": None,
            "train_loss_history": defaultdict(list),
            "test_loss_history": None,
            "current_epoch_id": 0,
            "current_batch_id": 0,
        }
        
        self.binary_criterion = torch.nn.BCELoss()
        self.multiclass_criterion = torch.nn.CrossEntropyLoss()
        self.reconstruction_criterion = torch.nn.MSELoss()
        self.l_contraction = l_contraction
        self.ignore_sample_zeros = ignore_sample_zeros
        self.optimizers = None
        self.model = None

    def hook(self, position, hooks):
        if position not in self.hooks:
            raise ValueError("Available hooks: %s" % self.hooks.keys())
        self.hooks[position].add(hooks)

    def _add_contraction(self):
        if self.l_contraction == 0:
            return 0.
        cont = 0.
        for k, v in self.model.contractables_grads.items():
            cont += torch.norm(v, 2)
        # if cont > 0:
            # cont = 1/cont 
        return self.l_contraction*cont

    def _train_supervised(self, predictions, targets, criterion, optimizer, contraction, ignore_sample_zeros):
        optimizer.zero_grad()
        if ignore_sample_zeros:
            idx = targets != 0
            predictions = predictions[idx] 
            targets = targets[idx] 

        loss = criterion(predictions, targets)
        if contraction:
            cont = self._add_contraction()
            loss += cont
        else :
            cont = 0.
        loss.backward()
        optimizer.step()
        return loss.item(), cont

    def _train_classifier(self, nb_class, predictions, targets, optimizer, contraction):
        criterion = self.multiclass_criterion
        return self._train_supervised(predictions, targets, criterion, optimizer, contraction, ignore_sample_zeros=False)

    def _train_condition_prediction(self, classifier, nb_class, reals, fakes, targets, p_optimizer, g_optimizer, train_p, train_g):
        criterion = self.multiclass_criterion
        
        p_loss = None
        if train_p:
            #TRAIN PREDICTOR ON REALS
            if p_optimizer:
                p_optimizer.zero_grad()
            
            pred_reals = classifier(reals)
            p_loss_reals = criterion(pred_reals, targets)
            
            if p_optimizer:
                p_loss_reals.backward()
                p_optimizer.step()
            p_loss = p_loss_reals.item()

        g_loss = None
        if train_g :
            #TRAIN GENERATOR
            if g_optimizer:
                g_optimizer.zero_grad()
            nz_fakes = fakes.detach()
            nz_fakes[reals == 0] = 0
            pred_fakes = classifier(nz_fakes)
            g_loss_fakes = criterion(pred_fakes, targets)
            
            if g_optimizer:
                g_loss_fakes.backward()
                g_optimizer.step()

            g_loss = g_loss_fakes.item()
        
        return p_loss, g_loss

    def _train_gan(self, model, discriminator, reals, fakes, run_device, g_optimizer, d_optimizer, train_g, train_d):
        targets_fake = torch.ones(reals.size(0), dtype=torch.float).view((-1, 1)).to(run_device)
        targets_real = torch.zeros_like(targets_fake, dtype=torch.float).view((-1, 1)).to(run_device)

        g_loss_val = None
        d_loss_val = None
        
        #TRAIN DISCRIMINATOR
        if train_d:
            if d_optimizer:
                d_optimizer.zero_grad()
    
            pred_fakes = discriminator(fakes.detach())
            pred_reals = discriminator(reals)
            if model.wgan:
                d_loss = -torch.mean(pred_reals) + torch.mean(pred_fakes)
            else :
                d_loss_fakes = self.binary_criterion(pred_fakes, targets_fake)
                d_loss_reals = self.binary_criterion(pred_reals, targets_real)
                d_loss = (d_loss_fakes + d_loss_reals) / 2
    
            if d_optimizer :       
                d_loss.backward()
                d_optimizer.step()

            if model.wgan:
                for param in model.output_gan_discriminator.parameters():
                    param.data.clamp_(-0.01, 0.01)

            d_loss_val = d_loss.item()
        
        #TRAIN GENERATOR
        if train_g:
            if g_optimizer:
                g_optimizer.zero_grad()
    
            pred = discriminator(fakes)
            if model.wgan:
                g_loss = -torch.mean(pred)
            else:
                g_loss = self.binary_criterion(pred, targets_real)
            
            if g_optimizer:
                g_loss.backward()
                g_optimizer.step()
            g_loss_val = g_loss.item()

        return g_loss_val, d_loss_val

    def train(
        self,
        model,
        train_loader,
        batch_formater,
        train_reconctruction_freq,
        train_condition_adv_freq,
        train_gan_discrimator_freq,
        train_gan_generator_freq,
        train_condition_fit_predictor_freq,
        train_condition_fit_generator_freq,
        cond_adv_sampling=False,
        cond_fitting_sampling=False,
        gan_sampling=False,
    ):
        
        run_device = model.run_device

        self.binary_criterion.to(run_device)
        self.multiclass_criterion.to(run_device)
        self.reconstruction_criterion.to(run_device)

        model.train()
        train_losses = defaultdict(list)
        self.meta["current_batch_id"] = 0
        for batch in train_loader:
            self.hooks["train_pass_starts"](model, self)

            samples, condition = batch_formater(batch)            
            samples = samples.to(run_device)
            condition = condition.to(run_device)

            if cond_adv_sampling or cond_fitting_sampling or gan_sampling:
                fiber_sample = torch.rand( size = (samples.size(0), model.fiber_space.out_dim) ).to(run_device)
                fiber_sample = (fiber_sample * 2) - 1
            
            #TRAIN RECONSTRUCTION   
            if train_reconctruction_freq > 0 and self.meta["current_batch_id"] % train_reconctruction_freq == 0:
                if self.optimizers["reconstruction"] is not None :
                    ret, cont = self._train_supervised(recons, samples, self.reconstruction_criterion, self.optimizers["reconstruction"], ignore_sample_zeros=self.ignore_sample_zeros, contraction=True)
                    train_losses["reconstruction"].append(ret)
                    if cont > 0:
                        train_losses["reconstruction_contraction"].append(cont)
        
            #TRAIN CONDITION ADVERSARIAL (DANN)   
            if cond_fitting_sampling:
                # fiber_sample = torch.rand( size = (samples.size(0), model.fiber_space.out_dim) ).to(run_device)
                # fiber_sample = (fiber_sample * 2) - 1
                recons = model.forward_output(fiber_sample, condition, fiber_input=True)
            else:
                recons = model.forward_output(samples, condition)
            
            if train_condition_adv_freq > 0 and self.meta["current_batch_id"] % train_condition_adv_freq == 0 :
                if self.optimizers["condition_adv"] is not None :
                    # model.forward_output(samples, condition)
                    condition_adv = model.predict_fiber_condition()
                    ret, cont = self._train_classifier(model.nb_class, condition_adv, condition, self.optimizers["condition_adv"], contraction=True)
                    train_losses["condition_adv"].append(ret)
                    if cont > 0:
                        train_losses["condition_adv_contraction"].append(cont)

            #TRAIN CONDITION FITING PREDICTOR   
            if gan_sampling:
                # fiber_sample = torch.rand( size = (samples.size(0), model.fiber_space.out_dim) ).to(run_device)
                # fiber_sample = (fiber_sample * 2) - 1
                recons = model.forward_output(fiber_sample, condition, fiber_input=True)
            else:
                recons = model.forward_output(samples, condition)
            p_loss, g_loss = self._train_condition_prediction(
                classifier=model.predict_condition,
                nb_class=model.nb_class,
                reals=samples,
                fakes=recons,
                targets=condition,
                p_optimizer=self.optimizers["prediction"],
                g_optimizer=self.optimizers["condition_fit_generator"],
                train_p = train_condition_fit_predictor_freq> 0 and (self.meta["current_batch_id"] % train_condition_fit_predictor_freq == 0),
                train_g = train_condition_fit_generator_freq > 0 and (self.meta["current_batch_id"] % train_condition_fit_generator_freq == 0)
            )
            if p_loss: train_losses["prediction"].append(p_loss)
            if g_loss: train_losses["condition_fit_generator"].append(g_loss)

            self.hooks["train_pass_stops"](model, self)
            self.meta["current_batch_id"] += 1
            
            #TRAIN GAN
            if gan_sampling:
                gan_fiber = torch.rand( size = (samples.size(0), model.fiber_space.out_dim) ).to(run_device)
                gan_fiber = (gan_fiber * 2) - 1
                gan_fakes = model.forward_output(gan_fiber, condition, fiber_input=True)
            else:
                gan_fakes = model.forward_output(samples, condition)
            g_loss, d_loss = self._train_gan(
                model,
                model.predict_gan,
                samples,
                gan_fakes,
                run_device,
                self.optimizers["gan_generator"],
                self.optimizers["gan_discriminator"],
                train_g = train_gan_generator_freq > 0 and (self.meta["current_batch_id"] % train_gan_generator_freq == 0),
                train_d = train_gan_discrimator_freq > 0 and (self.meta["current_batch_id"] % train_gan_discrimator_freq == 0)
            )
            if g_loss: train_losses["gan_generator"].append(g_loss)
            if d_loss: train_losses["gan_discriminator"].append(d_loss)

        for name in train_losses:
            average = torch.mean(torch.tensor(train_losses[name]))
            train_losses[name]=average

        return train_losses

    def test(self, model, test_loader, batch_formater, loss_obj):
        model.eval()
        test_loss= 0
        self.meta["current_batch_id"] = 0
        with torch.no_grad():
            for batch in test_loader:
                self.meta["current_batch_id"] += 1
                self.hooks["test_pass_starts"](model, self)
                data, target = batch_formater(batch)
                data = data.to(run_device)
                target = target.to(run_device)
              
                recons = model(data, target)
                loss = loss_obj(recons, data)
                loss.to(run_device)
                    
                self.batch_loss = loss.item()
                test_loss += self.batch_loss
                self.hooks["test_pass_stops"](model, self)
            
        test_loss /= len(test_loader.dataset)
        return test_loss

    def get_history(self):
        return {"train": self.meta["train_loss_history"], "test": self.meta["test_loss_history"]}

    def run(
        self,
        model,
        nb_epochs,
        train_loader,
        batch_formater,
        reconstruction_opt_fct,
        condition_adv_opt_fct,
        condition_fit_opt_fct,
        condition_fit_generator_opt_fct,
        gan_generator_opt_fct,
        gan_discriminator_opt_fct,
        train_reconctruction_freq=1,
        train_condition_adv_freq=1,
        train_gan_discrimator_freq=1,
        train_gan_generator_freq=1,
        train_condition_fit_predictor_freq=1,
        train_condition_fit_generator_freq=1,
        gan_sampling=False,
        test_loader=None
    ):
        from tqdm import trange
        
        pbar = trange(nb_epochs)
        self.optimizers = {
            "reconstruction": reconstruction_opt_fct(
                list(model.backbone.parameters()) + list(model.fiber.parameters()) + list(model.conditions.parameters())
            ),
            "condition_adv": condition_adv_opt_fct(
                list(model.fiber_condition_adv.parameters()) + list(model.fiber.parameters())
            ),
            "prediction": condition_fit_opt_fct(
                model.output_classifier.parameters()
            ),
            "condition_fit_generator": condition_fit_generator_opt_fct(
                list(model.backbone.parameters()) + list(model.conditions.parameters())
            ),
            "gan_generator": gan_generator_opt_fct( list(model.backbone.parameters()) + list(model.conditions.parameters())),
            "gan_discriminator": gan_discriminator_opt_fct(model.output_gan_discriminator.parameters())
        }
        self.model = model
        self.hooks["starts"](model, self)
        self.meta["current_epoch_id"] = 0
        try :
            for epoch in pbar:
                self.meta["current_epoch_id"] += 1
                self.hooks["epoch_starts"](model, self)
                train_loss = self.train(
                    model=model,
                    train_loader=train_loader,
                    batch_formater=batch_formater,
                    train_reconctruction_freq=train_reconctruction_freq,
                    train_condition_adv_freq=train_condition_adv_freq,
                    train_gan_discrimator_freq=train_gan_discrimator_freq,
                    train_gan_generator_freq=train_gan_generator_freq,
                    train_condition_fit_predictor_freq=train_condition_fit_predictor_freq,
                    train_condition_fit_generator_freq=train_condition_fit_generator_freq,
                    gan_sampling=gan_sampling
                )
                self.last_train_loss = train_loss
                label = ",".join(["%s: %.4f" % ("".join([ss[0]+ss[1] for ss in name.split("_")]), train_loss[name]) for name in train_loss])# * 1000
                pbar.set_description( label )
                for key in train_loss:
                    # print(key, train_loss[key])
                    self.meta["train_loss_history"][key].append(train_loss[key])
                if test_loader:
                    die
                    test_loss = self.test(model, test_loader, batch_formater, run_device)
                    self.last_test_loss = test_loss
                    self.meta["test_loss_history"].append(test_loss)
                self.hooks["epoch_stops"](model, self)
            self.hooks["stops"](model, self)
        except KeyboardInterrupt:
            print('Training interrupted!')
        
        return self.get_history()
