import torch
import torch.distributed as dist
import torch.optim as optim


def optimize(args, config, data_loader, data_loader_dd, travel_time):
    if (args.opt.lower() == "lbfgs") or (args.opt.lower() == "bfgs"):
        optimizer = optim.LBFGS(params=travel_time.parameters(), max_iter=1000, line_search_fn="strong_wolfe")
    elif args.opt.lower() == "adam":
        optimizer = optim.Adam(params=travel_time.parameters(), lr=0.1)
    elif args.opt.lower() == "sgd":
        optimizer = optim.SGD(params=travel_time.parameters(), lr=10.0)
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")

    # init loss
    loss = 0
    loss_dd = 0
    for meta in data_loader:
        station_index = meta["station_index"]
        event_index = meta["event_index"]
        phase_time = meta["phase_time"]
        phase_type = meta["phase_type"]
        phase_weight = meta["phase_weight"]
        loss += travel_time(
            station_index,
            event_index,
            phase_type,
            phase_time,
            phase_weight,
            double_difference=False,
        )["loss"]
        if args.distributed:
            dist.barrier()
            dist.all_reduce(loss)

        if args.double_difference:
            for meta in data_loader_dd:
                station_index = meta["station_index"]
                event_index = meta["event_index"]
                phase_time = meta["phase_time"]
                phase_type = meta["phase_type"]
                phase_weight = meta["phase_weight"]

                loss_dd += travel_time(
                    station_index,
                    event_index,
                    phase_type,
                    phase_time,
                    phase_weight,
                    double_difference=True,
                )["loss"]
                if args.distributed:
                    dist.barrier()
                    dist.all_reduce(loss_dd)

    print(f"Init loss: {loss+loss_dd}:  {loss} + {loss_dd}")

    if (args.opt.lower() == "lbfgs") or (args.opt.lower() == "bfgs"):
        prev_loss = 0
        for i in range(args.epochs):

            def closure():
                optimizer.zero_grad()
                for meta in data_loader:
                    station_index = meta["station_index"]
                    event_index = meta["event_index"]
                    phase_time = meta["phase_time"]
                    phase_type = meta["phase_type"]
                    phase_weight = meta["phase_weight"]
                    loss = travel_time(
                        station_index,
                        event_index,
                        phase_type,
                        phase_time,
                        phase_weight,
                        double_difference=False,
                    )["loss"]
                    if args.distributed:
                        dist.barrier()
                        dist.all_reduce(loss)
                    loss.backward()

                    if args.double_difference:
                        for meta in data_loader_dd:
                            station_index = meta["station_index"]
                            event_index = meta["event_index"]
                            phase_time = meta["phase_time"]
                            phase_type = meta["phase_type"]
                            phase_weight = meta["phase_weight"]

                            loss_dd = travel_time(
                                station_index,
                                event_index,
                                phase_type,
                                phase_time,
                                phase_weight,
                                double_difference=True,
                            )["loss"]
                            if args.distributed:
                                dist.barrier()
                                dist.all_reduce(loss_dd)

                            (loss_dd * args.dd_weight).backward()

                return loss

            optimizer.step(closure)

            loss = 0
            loss_dd = 0
            for meta in data_loader:
                station_index = meta["station_index"]
                event_index = meta["event_index"]
                phase_time = meta["phase_time"]
                phase_type = meta["phase_type"]
                phase_weight = meta["phase_weight"]
                loss += travel_time(
                    station_index,
                    event_index,
                    phase_type,
                    phase_time,
                    phase_weight,
                    double_difference=False,
                )["loss"]
            if args.double_difference:
                for meta in data_loader_dd:
                    station_index = meta["station_index"]
                    event_index = meta["event_index"]
                    phase_time = meta["phase_time"]
                    phase_type = meta["phase_type"]
                    phase_weight = meta["phase_weight"]

                    loss_dd = travel_time(
                        station_index,
                        event_index,
                        phase_type,
                        phase_time,
                        phase_weight,
                        double_difference=True,
                    )["loss"]
            if abs((loss + loss_dd) - prev_loss) < 1e-6:
                break
            prev_loss = loss + loss_dd
            print(f"Invert loss: {loss+loss_dd}:  {loss} + {loss_dd}")

            # set variable range
            travel_time.event_loc.weight.data[:, 2] += (
                (torch.rand_like(travel_time.event_loc.weight.data[:, 2]) - 0.5) * (args.epochs - 1 - i) / args.epochs
            )
            travel_time.event_loc.weight.data[:, 2].clamp_(min=config["zlim_km"][0], max=config["zlim_km"][1])
            # travel_time.event_loc.weight.data[:, 2].clamp_(min=config["mindepth"], max=config["maxdepth"])

    else:
        for i in range(args.epochs):
            optimizer.zero_grad()

            prev_loss = 0
            loss = 0
            loss_dd = 0
            for meta in data_loader:
                station_index = meta["station_index"]
                event_index = meta["event_index"]
                phase_time = meta["phase_time"]
                phase_type = meta["phase_type"]
                phase_weight = meta["phase_weight"]

                loss = travel_time(
                    station_index,
                    event_index,
                    phase_type,
                    phase_time,
                    phase_weight,
                    double_difference=False,
                )["loss"]
                loss.backward()

            if args.double_difference:
                for meta in data_loader_dd:
                    station_index = meta["station_index"]
                    event_index = meta["event_index"]
                    phase_time = meta["phase_time"]
                    phase_type = meta["phase_type"]
                    phase_weight = meta["phase_weight"]

                    loss_dd = travel_time(
                        station_index,
                        event_index,
                        phase_type,
                        phase_time,
                        phase_weight,
                        double_difference=True,
                    )["loss"]
                    (loss_dd * args.dd_weight).backward()

            optimizer.step()

            if abs((loss + loss_dd) - prev_loss) < 1e-3:
                print(f"Loss: {loss+loss_dd}:  {loss} + {loss_dd}")
                break
            prev_loss = loss + loss_dd

            if i % 100 == 0:
                print(f"Loss: {loss+loss_dd}:  {loss} + {loss_dd}")

            # set variable range
            travel_time.event_loc.weight.data[:, 2] += (
                torch.randn_like(travel_time.event_loc.weight.data[:, 2]) * (args.epochs - i) / args.epochs
            )
            travel_time.event_loc.weight.data[:, 2].clamp_(min=config["z(km)"][0], max=config["z(km)"][1])
