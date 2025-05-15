import numpy as np

semicircle=True
cancer=True


### SYNTHETIC SEMICIRCLE ###
if semicircle:
    circle_euclidean_no_potential = [
        0.3253518342971802,  # no_potential_4
        0.32164373993873596, # no_potential_0
        0.3180488348007202,  # no_potential_2
        0.3176177442073822,  # no_potential_3
        0.30934667587280273  # no_potential_1
    ]
    nll_euclidean_no_potential = [
        -40864.32139024061,  # no_potential_4
        -42212.907753407024, # no_potential_0
        -35681.20057505743,  # no_potential_2
        -41488.5114502541,   # no_potential_3
        -36036.630153507984  # no_potential_1
    ]

    circle_euclidean_w_potential = [
        0.14417582750320435, # w_potential_3
        0.12517841160297394, # w_potential_1
        0.12172325700521469, # w_potential_4
        0.1191103532910347,  # w_potential_0
        0.11612585186958313  # w_potential_2
    ]
    nll_euclidean_w_potential = [
        -4122.259827749953,   # w_potential_3
        -2438.0866718529087,  # w_potential_1
        -2490.748946320899,   # w_potential_4
        -1803.8457689373556,  # w_potential_0
        -2351.844211509739    # w_potential_2
    ]

    circle_learned_no_potential = [
        0.21726274490356445, # NLOT_3
        0.19580435752868652, # NLOT_4
        0.15148134529590607, # NLOT_0
        0.15054559707641602, # NLOT_1
        0.12424600124359131  # NLOT_2
    ]
    nll_learned_no_potential = [
        -14531.771008493788, # NLOT_3
        -13423.590471064956, # NLOT_4
        -5138.582481192413,  # NLOT_0
        -7160.787410565351,  # NLOT_1
        -2862.2237718380597  # NLOT_2
    ]

    circle_learned_w_potential = [
        0.10686294734477997, # ours_3
        0.10024629533290863, # ours_0
        0.09814503788948059, # ours_1
        0.0949002057313919,  # ours_2
        0.07774363458156586  # ours_4
    ]
    nll_learned_w_potential = [
        -1408.1058272320402, # ours_3
        -1172.655852419242,  # ours_0
        -999.7874291894074,  # ours_1
        -854.351464746798,   # ours_2
        -682.9412110213909   # ours_4
    ]


    # for each NLL, divide by 400 and make negative to get NLL per sample
    nll_euclidean_no_potential = -np.array(nll_euclidean_no_potential) / 400
    nll_euclidean_w_potential = -np.array(nll_euclidean_w_potential) / 400
    nll_learned_no_potential = -np.array(nll_learned_no_potential) / 400
    nll_learned_w_potential = -np.array(nll_learned_w_potential) / 400

    def print_results(results, name):
        mean = np.mean(results)
        std = np.std(results)
        ci = 1.96 * std / np.sqrt(len(results))
        print(f"{name}: {mean:.3f} ± {ci:.3f}")

    print("Synthetic Semicircle Results")
    print("Euclidean No Potential")
    print_results(circle_euclidean_no_potential, "Euclidean No Potential Circle Dist")
    print_results(nll_euclidean_no_potential, "Euclidean No Potential NLL")
    print("\nEuclidean With Potential")
    print_results(circle_euclidean_w_potential, "Euclidean With Potential Circle Dist")
    print_results(nll_euclidean_w_potential, "Euclidean With Potential NLL")
    print("\nLearned No Potential")
    print_results(circle_learned_no_potential, "Learned No Potential Circle Dist")
    print_results(nll_learned_no_potential, "Learned No Potential NLL")
    print("\nLearned With Potential")
    print_results(circle_learned_w_potential, "Learned With Potential Circle Dist")
    print_results(nll_learned_w_potential, "Learned With Potential NLL")


### CANCER ###
if cancer:
    leared_w_potential = [58.0591, 59.9051, 59.5262, 57.7009, 59.5101]

    learned_no_potential= [56.8460, 55.4440, 56.1040, 58.0044, 57.9242]

    eucl_w_potential = [27.3268, 28.3476, 24.8110, 26.2228, 26.9399]

    eucl_no_potential = [49.8550, 50.6025, 52.0376, 52.4575, 54.5995]

    print("\nCancer Results")
    print("Learned W Potential")
    print_results(leared_w_potential, "Learned W Potential Reward")
    print("Learned No Potential")
    print_results(learned_no_potential, "Learned No Potential Reward")
    print("Euclidean W Potential")
    print_results(eucl_w_potential, "Euclidean W Potential Reward")
    print("Euclidean No Potential")
    print_results(eucl_no_potential, "Euclidean No Potential Reward")