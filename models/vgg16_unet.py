import segmentation_models_pytorch as smp


def build_vgg16_unet():

    model = smp.Unet(

        encoder_name="vgg16",

        encoder_weights="imagenet",

        in_channels=1,

        classes=3

    )

    return model