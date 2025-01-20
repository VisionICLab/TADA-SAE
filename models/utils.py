import locale


def count_parameters(model):
    """
    Returns the number of trainable parameters in a model.
    From https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
    """
    locale.setlocale(locale.LC_ALL, "en_GB.UTF-8")
    num_params = sum(p.numel() for p in model.parameters())
    return locale.format_string("%d", num_params, True)

