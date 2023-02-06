
def get_params_func(opts, model, mode=False):
    
    # xxx Set up optimizer
    params = []
    if mode:
        print('get params for SSUL mode')
        if opts.step > 0:
            # freeze encoder and old classes output channel
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.cls[:2].parameters()),
                    'weight_decay': opts.weight_decay
                }
            )
            #BACKGROUND CHANNEL
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.cls[-1].parameters()),
                    'weight_decay': opts.weight_decay
                }
            )
            #NEW CLASS CHANNEL + UNKNOWN CHANNEL
        else:
            # not freeze, train entire network
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.body.parameters()),
                    'weight_decay': opts.weight_decay
                }
            )
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.head.parameters()),
                    'weight_decay': opts.weight_decay
                }
            )
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.cls.parameters()),
                    'weight_decay': opts.weight_decay
                }
            )
    else:
        print('get params for other mode')
        if not opts.freeze:
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.body.parameters()),
                    'weight_decay': opts.weight_decay
                }
            )

        params.append(
            {
                "params": filter(lambda p: p.requires_grad, model.head.parameters()),
                'weight_decay': opts.weight_decay
            }
        )

        if opts.lr_old is not None and opts.step > 0:
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.cls[:-1].parameters()),
                    'weight_decay': opts.weight_decay,
                    "lr": opts.lr_old * opts.lr
                }
            )
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.cls[-1:].parameters()),
                    'weight_decay': opts.weight_decay
                }
            )
        else:
            params.append(
                {
                    "params": filter(lambda p: p.requires_grad, model.cls.parameters()),
                    'weight_decay': opts.weight_decay
                }
            )
    if model.scalar is not None:
        params.append({"params": model.scalar, 'weight_decay': opts.weight_decay})

    return params