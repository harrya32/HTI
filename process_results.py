import numpy as np

semicircle=True
cancer=False


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

    circle_euclidean_w_potential_old = [
        0.14417582750320435, # w_potential_3
        0.12517841160297394, # w_potential_1
        0.12172325700521469, # w_potential_4
        0.1191103532910347,  # w_potential_0
        0.11612585186958313  # w_potential_2
    ]

    circle_euclidean_w_potential = [
        0.024325357750058174, 
        0.01795460283756256, 
        0.013586608693003654, 
        0.015443442389369011, 
        0.014468795619904995,
        0.013506781309843063, 
        0.014727648347616196, 
        0.011793147772550583, 
        0.012627781368792057, 
        0.016821635887026787, 
        0.020215362310409546, 
        0.013454342260956764, 
        0.010439550504088402, 
        0.015229828655719757, 
        0.013795467093586922, 
        0.010710522532463074, 
        0.012260863557457924, 
        0.03148188441991806, 
        0.021061623468995094, 
        0.013307000510394573
    ]

    nll_euclidean_w_potential = [-0.36060652567472373, -0.3931367408855143, -0.6703573429073519, -0.3371595009068954, -0.15318664744132296, -0.6759388454411664, -0.6471508625817244, -0.9582558673694015, -0.6056611382595108, -0.649404368431741, -0.41851248505161315, -0.7353171258405842, -0.8575138918731946, -0.49520025479545116, -0.743995188063081, -0.6702363707913996, -0.5480181698852125, 0.17007854640055975, -0.26508276102868994, -0.6182376544161282]


    nll_euclidean_w_potential_old = [
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

    circle_learned_w_potential_old = [
        0.10686294734477997, # ours_3
        0.10024629533290863, # ours_0
        0.09814503788948059, # ours_1
        0.0949002057313919,  # ours_2
        0.07774363458156586  # ours_4
    ]
    nll_learned_w_potential_old = [
        -1408.1058272320402, # ours_3
        -1172.655852419242,  # ours_0
        -999.7874291894074,  # ours_1
        -854.351464746798,   # ours_2
        -682.9412110213909   # ours_4
    ]

    circle_learned_w_potential = [0.024320702999830246, 0.01523534394800663, 0.013407470658421516, 0.03029140643775463, 0.012494776397943497, 0.015264466404914856, 0.011310990899801254, 0.02230796031653881, 0.01682858169078827, 0.014971684664487839, 0.014625245705246925, 0.01476896833628416, 0.011250115931034088, 0.017536476254463196, 0.018492694944143295, 0.014866560697555542, 0.013866539113223553, 0.011992167681455612, 0.009029998444020748, 0.01630055531859398]
    nll_learned_w_potential = [-0.44684341602467414, -0.663891750591091, -0.425360316326939, -0.32896318869418106, -0.5125377419893591, -0.5778815708746214, -0.89017990444372, -0.5700864323901196, -0.9419078184780223, -0.8412272256532791, -0.6223579420959536, -0.6659436427467357, -0.8726769545902913, -0.5411204007286374, -0.5134945232869359, -0.7864698321341463, -0.9354974653108812, -0.8430516402159376, -0.9691005291825308, -0.29758737353295694]

    circle_ablation_old = [
        0.12398120760917664,
        0.11444838345050812,
        0.11328815668821335,
        0.09702225774526596,
        0.09437358379364014
    ]

    nll_ablation_old = [
        -777.8702795935536,
        -1378.1410263173366,
        -1550.995868883021,
        -1716.6841440784901,
        -1800.00983051697
    ]

    circle_ablation = [0.02060532197356224, 0.009922867640852928, 0.014247261919081211, 0.014647409319877625, 0.012361662462353706, 0.013932624831795692, 0.014233006164431572, 0.016354620456695557, 0.02356737107038498, 0.016955053433775902, 0.015667645260691643, 0.01147887110710144, 0.015140872448682785, 0.017369117587804794, 0.012396465986967087, 0.017607389017939568, 0.016790036112070084, 0.0270828939974308, 0.013861404731869698, 0.022929459810256958]
    nll_ablation = [-0.3583183473031694, -0.7316764242813512, -0.7740320589580373, -0.32501374121028886, -0.5048991190541903, -0.5396012768981786, -0.4647470757851053, -0.6652391952424727, -0.36015236321294797, -0.7541969596111868, -0.6674968427593848, -0.6248220855187039, -0.7531053069964166, -0.7678158944217912, -0.7058290073560917, -0.4186512289449308, -0.585607438285364, -0.5466934340687614, -0.7751727491450588, -0.7071220065359068]

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
    print("\nAblation")
    print_results(circle_ablation, "Ablation Circle Dist")
    print_results(nll_ablation, "Ablation NLL")


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