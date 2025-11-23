import torch
import torch.nn as nn
import copy
import inspect
import logging
import timm

from backbone.vit_inflora import (
    VisionTransformer,
    PatchEmbed,
    Block,
    resolve_pretrained_cfg,
    build_model_with_cfg,
    checkpoint_filter_fn,
    _get_pretrained_cfg_value,
)

class ViT_lora_co(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, n_tasks=n_tasks, rank=rank)


    def forward(self, x, task_id, register_blk=-1, get_feat=False, get_cur_feat=False):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)
        for i, blk in enumerate(self.blocks):
            x = blk(x, task_id, register_blk==i, get_feat=get_feat, get_cur_feat=get_cur_feat)

        x = self.norm(x)
        
        return x, prompt_loss



def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # Create model without pretrained weights first
    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = _get_pretrained_cfg_value(pretrained_cfg, 'num_classes', kwargs.get('num_classes', 1000))
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    build_args = dict(
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
    )
    
    # Create the custom model
    model = build_model_with_cfg(
        ViT_lora_co, variant, False,  # Always create without pretrained first
        **build_args,
        **kwargs)
    
    # Load pretrained weights using timm if requested
    if pretrained:
        try:
            logging.info(f"Loading pretrained weights for {variant} using timm...")
            # The variant is registered in vit_inflora.py which causes parameter conflicts
            # Solution: use timm's model factory with source parameter if available,
            # or use a workaround to bypass our registration
            
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Method 1: Try using timm's create_model with source='hf_hub' or similar
                # to bypass local registrations (if supported)
                try:
                    # Check if timm supports source parameter
                    import inspect
                    create_model_sig = inspect.signature(timm.create_model)
                    if 'source' in create_model_sig.parameters:
                        # Try to use a source that bypasses local registry
                        pretrained_model = timm.create_model(
                            variant,
                            pretrained=True,
                            num_classes=0,
                            source='timm'  # Force using timm's native implementation
                        )
                    else:
                        # Method 2: Try to access timm's internal registry and temporarily modify it
                        # This is version-dependent, so we'll try a different approach
                        raise AttributeError("source parameter not available")
                except (AttributeError, TypeError, Exception) as e1:
                    # Method 3: Directly import timm's native vision_transformer function
                    # This bypasses the registry system entirely
                    try:
                        if variant == 'vit_base_patch16_224_in21k':
                            # Directly import the native function from timm
                            from timm.models.vision_transformer import vit_base_patch16_224_in21k as timm_fn
                            pretrained_model = timm_fn(pretrained=True, num_classes=0)
                            logging.info("Successfully loaded using direct import from timm.models.vision_transformer")
                        else:
                            raise AttributeError(f"Direct import not implemented for {variant}")
                    except (ImportError, AttributeError) as e2:
                        # Method 4: Try using vision_transformer module
                        try:
                            from timm.models import vision_transformer as timm_vit
                            if hasattr(timm_vit, variant):
                                model_fn = getattr(timm_vit, variant)
                                pretrained_model = model_fn(pretrained=True, num_classes=0)
                            else:
                                raise AttributeError(f"{variant} not found in vision_transformer")
                        except Exception as e3:
                            # Method 5: Last resort - use a similar unregistered model
                            # For in21k models, we can try the standard variant
                            logging.warning(f"Direct loading failed ({e1}, {e2}, {e3}), trying fallback...")
                            if 'in21k' in variant:
                                # Try standard variant (not registered, should work)
                                fallback_variant = variant.replace('_in21k', '')
                                logging.info(f"Using fallback variant: {fallback_variant}")
                                pretrained_model = timm.create_model(
                                    fallback_variant,
                                    pretrained=True,
                                    num_classes=0
                                )
                                logging.warning(f"Loaded {fallback_variant} instead of {variant}")
                            else:
                                # If all else fails, try the variant anyway and catch the error
                                pretrained_model = timm.create_model(
                                    variant,
                                    pretrained=True,
                                    num_classes=0
                                )
            
            pretrained_state_dict = pretrained_model.state_dict()
            del pretrained_model  # Free memory
            
            # Filter out keys that don't exist in our custom model (like LoRA parameters)
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            for key, value in pretrained_state_dict.items():
                # Skip head and classifier weights
                if 'head' in key or 'classifier' in key:
                    continue
                # Only include keys that exist in our model
                if key in model_state_dict:
                    # Check if shapes match
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        logging.debug(f"Skipping {key} due to shape mismatch: {model_state_dict[key].shape} vs {value.shape}")
            
            # Load the filtered state dict
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            if missing_keys:
                # Filter out LoRA-related keys as they're expected to be missing
                lora_missing = [k for k in missing_keys if 'lora' in k.lower()]
                other_missing = [k for k in missing_keys if 'lora' not in k.lower()]
                if other_missing:
                    logging.warning(f"Missing keys (non-LoRA): {other_missing[:5]}...")  # Show first 5
                if lora_missing:
                    logging.debug(f"Missing LoRA keys (expected): {len(lora_missing)} keys")
            if unexpected_keys:
                logging.debug(f"Unexpected keys (will be ignored): {len(unexpected_keys)} keys")
            
            loaded_count = len(filtered_state_dict)
            total_count = len([k for k in pretrained_state_dict.keys() if 'head' not in k and 'classifier' not in k])
            logging.info(f"Successfully loaded {loaded_count}/{total_count} pretrained weights for {variant}")
        except Exception as err:
            import traceback
            error_msg = str(err)
            logging.error(
                f"Failed to load pretrained weights for {variant}: {error_msg}\n"
                f"Traceback: {traceback.format_exc()}\n"
                f"Continuing with randomly initialized weights."
            )
            # Continue with randomly initialized model
    
    return model



class SiNet(nn.Module):

    def __init__(self, args):
        super(SiNet, self).__init__()

        # Determine the variant and pretrained flag from args
        # Default to in21k variant (original behavior) but allow override via config
        backbone_type = args.get("backbone_type", "vit_base_patch16_224")
        pretrained = args.get("pretrained", True)
        
        # Map backbone_type to timm variant
        # Default to in21k for better pretrained weights (original code behavior)
        if "in21k" in backbone_type.lower():
            variant = "vit_base_patch16_224_in21k"
        elif backbone_type == "vit_base_patch16_224":
            # Original code used in21k, so we'll use that for compatibility
            # But log a warning if config specifies standard variant
            variant = "vit_base_patch16_224_in21k"
            if backbone_type != "vit_base_patch16_224_in21k":
                logging.info(f"Using vit_base_patch16_224_in21k instead of {backbone_type} for better pretrained weights")
        else:
            variant = "vit_base_patch16_224_in21k"  # Default fallback
        
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, n_tasks=args["total_sessions"], rank=args["rank"])
        logging.info(f"Creating SiNet with variant={variant}, pretrained={pretrained}")
        self.image_encoder = _create_vision_transformer(variant, pretrained=pretrained, **model_kwargs)
        # print(self.image_encoder)
        # exit()

        self.class_num = 1
        self.class_num = args["init_cls"]
        self.classifier_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        self.classifier_pool_backup = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        # self.prompt_pool = CodaPrompt(args["embd_dim"], args["total_sessions"], args["prompt_param"])

        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image, task=None):
        if task == None:
            image_features, _ = self.image_encoder(image, self.numtask-1)
        else:
            image_features, _ = self.image_encoder(image, task)
        image_features = image_features[:,0,:]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image, get_feat=False, get_cur_feat=False, fc_only=False):
        if fc_only:
            fc_outs = []
            for ti in range(self.numtask):
                fc_out = self.classifier_pool[ti](image)
                fc_outs.append(fc_out)
            return torch.cat(fc_outs, dim=1)

        logits = []
        image_features, prompt_loss = self.image_encoder(image, task_id=self.numtask-1, get_feat=get_feat, get_cur_feat=get_cur_feat)
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)
        for prompts in [self.classifier_pool[self.numtask-1]]:
            logits.append(prompts(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
            'prompt_loss': prompt_loss
        }

    def interface(self, image, task_id = None):
        image_features, _ = self.image_encoder(image, task_id=self.numtask-1 if task_id is None else task_id)

        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits,1)
        return logits
    
    def interface1(self, image, task_ids):
        logits = []
        for index in range(len(task_ids)):
            image_features, _ = self.image_encoder(image[index:index+1], task_id=task_ids[index].item())
            image_features = image_features[:,0,:]
            image_features = image_features.view(image_features.size(0),-1)

            logits.append(self.classifier_pool_backup[task_ids[index].item()](image_features))

        logits = torch.cat(logits,0)
        return logits

    def interface2(self, image_features):

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits,1)
        return logits

    def update_fc(self, nb_classes):
        self.numtask +=1

    def classifier_backup(self, task_id):
        self.classifier_pool_backup[task_id].load_state_dict(self.classifier_pool[task_id].state_dict())

    def classifier_recall(self):
        self.classifier_pool.load_state_dict(self.old_state_dict)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
