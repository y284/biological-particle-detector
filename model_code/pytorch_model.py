import torch
import torch.nn as nn
import torch.nn.functional as F

class dotdict(dict):
    """A dictionary subclass that allows dot notation access to its keys.
    
    This extends Python's built-in dict to support attribute-style access (obj.key)
    in addition to normal dictionary access (obj['key']). This provides convenience
    when working with configuration dictionaries or data structures.

    Example:
        >>> d = dotdict({'key': 'value'})
        >>> d.key     # 'value' (dot access)
        >>> d['key']  # 'value' (standard dict access)
    """

    def __setattr__(self, name, value):
        """Allows setting attributes using dot notation.
        
        Args:
            name (str): The key/attribute name to set.
            value: The value to assign to the key/attribute.
        """
        # Redirect attribute assignment to dictionary key assignment
        dict.__setitem__(self, name, value)

    def __delattr__(self, name):
        """Allows deleting attributes using dot notation.
        
        Args:
            name (str): The key/attribute name to delete.
        """
        # Redirect attribute deletion to dictionary key deletion
        dict.__delitem__(self, name)

    def __getattr__(self, name):
        """Allows getting attributes using dot notation.
        
        Args:
            name (str): The key/attribute name to retrieve.

        Returns:
            The value associated with the given key.

        Raises:
            AttributeError: If the key does not exist in the dictionary.
        """
        try:
            # Try to access the key using normal dictionary lookup
            return self[name]
        except KeyError:
            # Convert KeyError to AttributeError for better semantics in attribute access
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

class ConvBNReLU3D(nn.Module):
    """A 3D convolutional block consisting of Conv3D + BatchNorm + ReLU activation.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Default: 3.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int, optional): Zero-padding added to all sides. Default: 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU3D, self).__init__()
        
        # Define a sequential block combining Conv3D, BatchNorm, and ReLU
        self.block = nn.Sequential(
            # 3D convolution layer (no bias since BatchNorm handles it)
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                bias=False  # Disable bias as BatchNorm will compensate
            ),
            
            # Batch normalization for stable training
            nn.BatchNorm3d(out_channels),
            
            # ReLU activation (inplace=True saves memory)
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass through the block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, depth, height, width).
            
        Returns:
            torch.Tensor: Output tensor after Conv3D → BatchNorm → ReLU.
        """
        return self.block(x)

class EncoderBlock(nn.Module):
    """A 3D encoder block with optional upsampling and downsampling capabilities.
    
    This block processes 3D volumetric data through convolutional layers and can optionally
    perform upsampling and downsampling. It supports skip connections and different upsampling methods.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        skip_channels (int, optional): Number of channels from skip connection. Default: 0.
        num_conv3d (int, optional): Number of ConvBNReLU3D layers. Default: 1.
        do_up (bool, optional): Whether to include upsampling. Default: True.
        do_down (bool, optional): Whether to include downsampling. Default: True.
        use_transpose (bool, optional): Whether to use transposed convolution for upsampling 
                                       instead of interpolation. Default: False.
    """
    
    def __init__(self, in_channels, out_channels, skip_channels=0, num_conv3d=1, 
                 do_up=True, do_down=True, use_transpose=False):
        super(EncoderBlock, self).__init__()
        
        # Upsampling configuration
        self.do_up = do_up
        if self.do_up:
            if use_transpose:
                # Learnable upsampling with transposed convolution
                self.upsample = nn.Sequential(
                    nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                # Simple trilinear interpolation upsampling
                self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='trilinear')
        
        # Downsampling configuration
        self.do_down = do_down
        if self.do_down:
            # Fixed trilinear interpolation downsampling
            self.downsample = lambda x: F.interpolate(x, scale_factor=0.5, mode='trilinear')
        
        # Main convolutional layers
        self.conv3d_layers = nn.Sequential(
            *[ConvBNReLU3D(
                # First layer gets input+skip channels, others get output channels
                out_channels if i != 0 else in_channels + skip_channels,
                out_channels,
                stride=(1, 1, 1)) 
              for i in range(num_conv3d)]
        )

    def forward(self, x, xskip=None):
        """Forward pass through the encoder block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, D, H, W).
            xskip (torch.Tensor, optional): Skip connection tensor to concatenate. Default: None.
            
        Returns:
            dotdict: A dictionary-like object containing:
                - out: The main output tensor
                - up: Upsampled output (if do_up=True)
                - down: Downsampled output (if do_down=True)
        """
        # Handle skip connection if provided
        if xskip is not None:
            x = torch.cat([x, xskip], dim=1)  # Concatenate along channel dimension
        
        # Process through convolutional layers
        out = self.conv3d_layers(x)
        
        # Prepare output dictionary
        output = {
            "out": out,    # Main output
            "up": None,    # Will contain upsampled output if enabled
            "down": None   # Will contain downsampled output if enabled
        }
        
        # Apply upsampling if enabled
        if self.do_up:
            output["up"] = self.upsample(out)
            
        # Apply downsampling if enabled
        if self.do_down:
            output["down"] = self.downsample(out)
            
        # Return as dotdict for attribute-style access
        return dotdict(output)

class DeepFinder(nn.Module):
    """A 3D convolutional neural network for volumetric segmentation tasks.
    
    The model consists of an encoder-decoder architecture with skip connections,
    designed for processing 3D data (e.g., medical imaging, particle detection).
    
    Args:
        channels (list, optional): List of channel sizes for each stage of the network.
                                 Default: [28, 32, 36].
    """
    
    def __init__(self, channels=[28, 32, 36]):
        super(DeepFinder, self).__init__()
        
        # Register a buffer for device tracking
        self.register_buffer('D', torch.tensor(0))
        
        # Define the types of outputs the model can produce
        self.output_type = ['particle'] # ['particle', 'loss'] for training
        
        # Input normalization layer
        self.norm = nn.BatchNorm3d(1)  # Normalize single-channel 3D input
        
        # Encoder blocks (downsampling pathway)
        self.encoder1 = EncoderBlock(
            in_channels=1,              # Input channel (3D volume)
            out_channels=channels[0],   # First expansion
            num_conv3d=2,               # Two convolutional layers
            do_up=False,                # No upsampling in first block
            do_down=True               # Include downsampling
        )
        
        self.encoder2 = EncoderBlock(
            in_channels=channels[0],   # Takes previous block's output
            out_channels=channels[1],  # Further channel expansion
            num_conv3d=2,              # Two convolutional layers
            do_up=False,                # No upsampling
            do_down=True               # Include downsampling
        )
        
        # Decoder blocks (upsampling pathway)
        self.decoder1 = EncoderBlock(
            in_channels=channels[1],   # Takes encoder2's downsampled output
            out_channels=channels[2],   # Highest channel dimension
            num_conv3d=4,              # Four convolutional layers for richer features
            do_up=True,                # Include upsampling
            do_down=False              # No further downsampling
        )
        
        self.decoder2 = EncoderBlock(
            in_channels=channels[2],   # Takes decoder1's output
            out_channels=channels[1],   # Matching encoder2's channels
            skip_channels=channels[1],  # Skip connection from encoder2
            num_conv3d=2,              # Two convolutional layers
            do_up=True,                # Include upsampling
            do_down=False,             # No downsampling
            use_transpose=True         # Use transposed conv for learnable upsampling
        )
        
        # Final preprocessing block before mask prediction
        self.pre = EncoderBlock(
            in_channels=channels[1],   # Takes decoder2's output
            out_channels=channels[0],  # Matching encoder1's channels
            num_conv3d=2,             # Two convolutional layers
            do_up=False,              # No upsampling
            do_down=False             # No downsampling
        )
        
        # Final 1x1x1 convolution to produce class logits
        self.mask = nn.Conv3d(
            in_channels=channels[0],  # Takes pre's output
            out_channels=6,           # 6 output classes (particle types)
            kernel_size=1,             # 1x1x1 convolution
            stride=1,
            bias=False                # No bias needed before softmax
        )

    def forward(self, batch):
        """Forward pass through the network.
        
        Args:
            batch (dict): Input dictionary containing:
                - 'volume': Input 3D tensor (batch_size, D, H, W)
                - 'label': Ground truth labels (optional, for training)
                
        Returns:
            dict: Output dictionary containing:
                - 'particle': Class probabilities (if in output_type)
                - 'loss': Cross-entropy loss (if in output_type and labels provided)
        """
        # Get device and prepare input volume
        device = self.D.device
        volume = batch["volume"].to(device).unsqueeze(1)  # Add channel dimension
        
        # Normalize input
        input_ = self.norm(volume)
        
        # Encoder pathway
        encode1 = self.encoder1(input_)       # First encoding stage
        encode2 = self.encoder2(encode1.down) # Second encoding stage (using downsampled output)
        
        # Decoder pathway with skip connections
        decode1 = self.decoder1(encode2.down) # First decoding stage (from encoder2's downsampled output)
        decode2 = self.decoder2(encode2.out,  # Second decoding stage with skip connection
                               decode1.up)    # from encoder2's main output
        
        # Final preprocessing
        pre = self.pre(decode2.up)            # Process decoder output
        
        # Generate final logits
        logit = self.mask(pre.out)            # 6-class prediction
        
        # Prepare output dictionary
        output = {}
        
        # Calculate loss if requested and labels available
        if "loss" in self.output_type and "label" in batch.keys():
            output["loss"] = F.cross_entropy(
                logit, 
                batch['label'].to(device), 
                label_smoothing=0.01,  # Regularization
            )
        
        # Generate particle predictions if requested
        if "particle" in self.output_type:
            output['particle'] = F.softmax(logit, 1)  # Class probabilities
        
        return output

class EnsembleTTADeepFinder(nn.Module):
    """An ensemble model with Test-Time Augmentation (TTA) capabilities.
    
    This model combines predictions from multiple models and applies test-time augmentation
    by averaging predictions across multiple geometric transformations of the input.
    
    Args:
        models (list): List of trained nn.Module instances to ensemble.
    """
    
    def __init__(self, models = [DeepFinder() for _ in range (4)]):
        super(EnsembleTTADeepFinder, self).__init__()
        # Store the individual models in a ModuleList for proper parameter handling
        self.models = nn.ModuleList(models)

    def forward(self, batch):
        """Forward pass with test-time augmentation.
        
        Args:
            batch (dict): Input dictionary containing:
                - 'volume': Input 3D tensor (batch_size, D, H, W)
                
        Returns:
            dict: Output dictionary containing:
                - 'particle': Averaged class probabilities after TTA
        """
        output = {"particle": 0}  # Initialize output container
        
        # Prepare original volume and get batch size
        volume = batch["volume"]
        b = len(volume)  # Batch size
        
        # Generate 7 augmented versions of the input (original + 6 transformations):
        # 1. Original volume
        # 2. Flip along z-axis (depth)
        # 3. Flip along y-axis (height)
        # 4. Flip along x-axis (width)
        # 5. 90° rotation
        # 6. 180° rotation
        # 7. 270° rotation
        z_flip = torch.flip(volume, [1])  # Depth flip
        y_flip = torch.flip(volume, [2])  # Height flip
        x_flip = torch.flip(volume, [3])  # Width flip
        rot_1 = torch.rot90(volume, k=1, dims=(-1, -2))  # 90° rotation
        rot_2 = torch.rot90(volume, k=2, dims=(-1, -2))  # 180° rotation
        rot_3 = torch.rot90(volume, k=3, dims=(-1, -2))  # 270° rotation
        
        # Combine all augmented versions into one large batch
        batch["volume"] = torch.cat([
            volume,    # Original
            x_flip,    # Flipped width
            y_flip,    # Flipped height
            z_flip,    # Flipped depth
            rot_1,     # Rotated 90°
            rot_2,     # Rotated 180°
            rot_3      # Rotated 270°
        ], dim=0)
        
        # Get predictions from all models in the ensemble
        all_predictions = 0
        for model in self.models:
            # Average predictions across models
            all_predictions += model(batch)["particle"]
        all_predictions /= len(self.models)
        
        # Reverse transformations and average predictions
        for i in range(7):  # For each augmentation
            current_pred = all_predictions[b*i:b*(i+1)]  # Get predictions for this augmentation
            
            if i == 0:
                # Original - no transformation needed
                output['particle'] += current_pred
            elif i < 4:
                # Flip predictions back (1-3 are flips)
                output['particle'] += torch.flip(current_pred, dims=[-i])
            else:
                # Rotate predictions back (4-6 are rotations)
                rot = i - 3  # Number of 90° rotations to undo (1, 2, or 3)
                output['particle'] += torch.rot90(current_pred, k=rot, dims=(-2, -1))
        
        # Average across all augmentations
        output['particle'] /= 7
        
        return output