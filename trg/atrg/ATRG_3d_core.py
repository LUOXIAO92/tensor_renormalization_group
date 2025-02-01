



def leg_transposition(A:Tensor, do_what="transpose", direction="t"):
    """
    A: tensor A_{TXYtxy} = U_{TXYi} s_i VH_{itxy} 
    >>>        T  Y
    >>>        | /
    >>>        U -- X
    >>>        s
    >>>   x -- VH
    >>>      / |
    >>>     y  t
    
    do_what: "transpose" or "restore"
    direction: "t" or "T", temporal direction; 
               "x" or "X", x direction; 
               "y" or "Y", y direction. 
    -------------------------------------------------------------
    >>> "transpose":
    >>>     "t": U_{TXYi} s_i VH_{itxy} -> U_{TXYi} s_i VH_{itxy}
    >>>     "x": U_{TXYi} s_i VH_{itxy} -> U_{XYTi} s_i VH_{ixyt}
    >>>     "y": U_{TXYi} s_i VH_{itxy} -> U_{YTXi} s_i VH_{iytx}
    >>> "restore":
    >>>     "t": U_{TXYi} s_i VH_{itxy} -> U_{TXYi} s_i VH_{itxy}
    >>>     "x": U_{XYTi} s_i VH_{ixyt} -> U_{TXYi} s_i VH_{itxy}
    >>>     "y": U_{YTXi} s_i VH_{iytx} -> U_{TXYi} s_i VH_{itxy}
    -------------------------------------------------------------
    """

    if do_what == "transpose":
        if   direction == "x" or direction == "X":
            A.U  = cp.transpose(A.U , axes=(1,2,0,3))
            A.VH = cp.transpose(A.VH, axes=(0,2,3,1))
        elif direction == "y" or direction == "Y":
            A.U  = cp.transpose(A.U , axes=(2,0,1,3))
            A.VH = cp.transpose(A.VH, axes=(0,3,1,2))
        elif direction == "t" or direction == "T":
            return A

    elif do_what == "restore":
        if   direction == "x" or direction == "X":
            A.U  = cp.transpose(A.U , axes=(2,0,1,3))
            A.VH = cp.transpose(A.VH, axes=(0,3,1,2))
        elif direction == "y" or direction == "Y":
            A.U  = cp.transpose(A.U , axes=(1,2,0,3))
            A.VH = cp.transpose(A.VH, axes=(0,2,3,1))
        elif direction == "t" or direction == "T":
            return A

    return A