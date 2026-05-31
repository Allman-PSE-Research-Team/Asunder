"""Load-balancing specific modules in Asunder."""

def LoadBalancer(*args, **kwargs):
    """
    Run benchmark evaluations using :mod:`asunder.load_balancing.column_generation.LB`.
    
    Parameters
    ----------
    *args : Any
        Additional positional arguments.
    **kwargs : Any
        Additional keyword arguments.
    
    Returns
    -------
    Any
        Computed result.
    """
    from asunder.load_balancing.column_generation.LB import LoadBalancer as _LoadBalancer

    return _LoadBalancer(*args, **kwargs)



__all__ = [
    "algorithms",
    "column_generation",
    "utils",
    "LoadBalancer",
]
