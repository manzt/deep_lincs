import six

def z_score(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def standard_scale(X):
    X = X - X.min(axis=0)
    return X / X.max(axis=0)


_GLOBAL_NORMALIZERS = {"z_score": z_score, "standard_scale": standard_scale}


def deserialize(identifier):
    if identifier in _GLOBAL_NORMALIZERS:
        fn = _GLOBAL_NORMALIZERS[identifier]
        return fn
    else:
        raise ValueError(f"Could not interpret normalization method: {identifier}")


def get_norm_method(identifier):
    """Get the `identifier` normalizer function.
    # Arguments
        identifier: None or str, name of the function.
    # Returns
        Normalization function
    # Raises
        ValueError if unknown identifier
    """
    if identifier is None:
        return linear
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError(f"Could not interpret normalization method: {identifer}")
