import os
import arg_parser
import torch
import torch.optim
import torch.utils.data
import statistics


def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")


    acc_retain = []
    acc_forget = []
    acc_test = []
    MIA = []
    SVC_MIA = []
    relearn_ep = []
    for seed in [1, 2, 3, 4]:
        root = './_results/celeba/seed' + str(seed)
        filepath = os.path.join(root, args.save_dir)

        checkpoint = torch.load(filepath, map_location=device)
        evaluation_result = checkpoint.get("evaluation_result")
        if checkpoint is not None:
            evaluation_result = checkpoint
            acc_retain.append(evaluation_result['accuracy']['retain'])
            acc_forget.append(evaluation_result['accuracy']['forget'])
            acc_test.append(evaluation_result['accuracy']['test'])
            MIA.append(evaluation_result['MIA'])
            SVC_MIA.append(evaluation_result["SVC_MIA_forget_efficacy"]['confidence'])

            print(evaluation_result['accuracy']['retain'], evaluation_result['accuracy']['forget'], evaluation_result['accuracy']['test'],
                  evaluation_result['MIA'], evaluation_result["SVC_MIA_forget_efficacy"]['confidence'])

            # relearn_ep.append(evaluation_result['relearn_ep'])
            # print(evaluation_result['relearn_ep'])

        else:
            print("No evaluation_result found")

    print(f"Accuracy Retain: {statistics.mean(acc_retain)}, {statistics.stdev(acc_retain)}")
    print(f"Accuracy Forget: {statistics.mean(acc_forget)}, {statistics.stdev(acc_forget)}")
    print(f"Accuracy Test: {statistics.mean(acc_test)}, {statistics.stdev(acc_test)}")
    print(f"MIA: {statistics.mean(MIA)}, {statistics.stdev(MIA)}")
    print(f"SVC MIA: {statistics.mean(SVC_MIA)}, {statistics.stdev(SVC_MIA)}")

    # print(f'Relearn EP: {statistics.mean(relearn_ep)}, {statistics.stdev(relearn_ep)}')


if __name__ == "__main__":
    main()


