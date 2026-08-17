# Definitions and needed quantities / constants

import numpy as np

# tank dimensions:
radius = 1.524
half_height = 1.98

# NCQE FV definition
FV_radius = 0.7
FV_half_height = 1.0

MC_POT_Empirical = 4.41733e20      # GENIE 0-16
MC_POT_Base = 4.426e20             # Base GENIE

def select_MC():
    which_MC = input(
        "\nPlease specify which MC sample:\n"
        "  1 = Empirical model\n"
        "  2 = Base GENIE\n"
        "Enter your selection (1 or 2): "
    ).strip()

    # Validate the MC sample selection
    if which_MC not in ('1', '2'):
        print(
            "\n" + 80*"#" + "\n"
            "Error!!! Please specify '1' (Empirical) or '2' (Base) "
            "and re-run the script\n"
            + 80*"#" + "\n"
        )
        exit()

    # Assign the appropriate POT based on the selected MC sample
    if which_MC == '1':
        POT = MC_POT_Empirical
        print('\n******* Empirical Model Selected *******\n')
    elif which_MC == '2':
        POT = MC_POT_Base
        print('\n******* Base GENIE Selected *******\n')

    return which_MC, POT
    

# Bunch timing properties
HALF_BUNCH_T = 18.936/2            # intrinsic BNB bunch interval
bunch_sigma = 3.5                  # derived bunch sigma for NCQE events

bunch_offset_mapping = 0.785       # offset mapping from MC CC to NCQE

# MC extracted CCinc bunch centers (https://github.com/S81D/BNB_bunches)
centers = np.array([31.080241331783608, 49.86887398044972, 68.93296122138285, 87.99776529989828, 106.92835749210317, 125.6744929249585, 144.55074229641292, 163.55533862061978, 182.42054838303565, 201.3975667029726, 220.45443080446125, 239.35228866298255, 258.3293431484236, 277.16438785242855, 296.09720802748456, 315.07114394995256, 334.03324394912823, 352.9587576490855, 371.8466462675708, 390.79194334882425, 409.63044291467685, 428.65567092584, 447.6549629320083, 466.57352377409313, 485.51073635902037, 504.4792057236341, 523.4956574718212, 542.3691320377058, 561.1099950563898, 580.2327318476815, 599.219010807661, 618.1019306658702, 636.9854167698654, 655.951836439576, 674.9453451130016, 693.8124506487408, 712.671642937589, 731.7183509540064, 750.6217703640303, 769.543817927331, 788.6698605766949, 807.5171007720303, 826.3570279447015, 845.2861546004002, 864.1547464834093, 883.1283566641179, 902.1248645210509, 920.9586744177927, 939.8944893020667, 958.8604582494381, 977.9115116970311, 996.7815728120514, 1015.6916816004485, 1034.593675449364, 1053.6218603405582, 1072.4974885887596, 1091.536793059606, 1110.3787443226672, 1129.5039961808134, 1148.208660322583, 1167.3275787742382, 1186.2991638425285, 1204.9871507926803, 1224.054624008908, 1242.917134430439, 1261.9481194256011, 1280.8989737693325, 1299.7479318121127, 1318.6949768605764, 1337.6816835239597, 1356.5741533701105, 1375.6189574036052, 1394.2645061208234, 1413.405519458969, 1432.3530855833285, 1451.3251805522368, 1470.3698038901168, 1489.1695033456674, 1508.0806190832418, 1526.995246745324, 1546.0600858753075])
bunch_time_start = centers[0] - bunch_offset_mapping - bunch_sigma
bunch_time_end = centers[-1] - bunch_offset_mapping + bunch_sigma
# ^ the spill start (end) times represent a - (+) 1-sigma cut from the first (last) bunch

# spill width for data
centers_data = np.array([196.56224433440246, 215.66909099116478, 234.50541331974833, 252.77573081775793, 271.82317272128086, 290.5525519751267, 309.47529727366754, 328.30485229514017, 347.30963148293534, 366.2956121865753, 385.226842076395, 404.22901923484017, 423.1284015795741, 442.1028957843268, 460.9563022567543, 479.9292098671415, 498.67523613302967, 517.745556275794, 536.5634329571434, 555.5826172158363, 574.5584125542986, 593.3412403321598, 612.4668820959361, 631.1017157126897, 650.245272991258, 669.1190067717141, 688.0583663595144, 707.0581456469093, 726.0269162795082, 744.773346737428, 763.7192011932474, 782.8556702712547, 801.7117975261683, 820.6641961303449, 839.4297222624705, 858.4883851813762, 877.452957835381, 896.3449195713947, 915.2971258549227, 934.355709628352, 953.1996719890523, 972.2816369135228, 991.1165854922059, 1009.9643936927976, 1028.951234108228, 1047.9972672500965, 1066.8793045738905, 1085.8351450523521, 1104.763384381811, 1123.7468193612021, 1142.803255690862, 1161.5532573658047, 1180.4810843925607, 1199.5057688888362, 1218.302045446269, 1237.1624520071487, 1256.1779033910852, 1275.2306259397883, 1294.0361634248936, 1313.1200477268103, 1331.9742215964689, 1350.8189134298868, 1369.786773197136, 1388.6371303935912, 1407.6174152883736, 1426.6042835718315, 1445.7252579081544, 1464.4532145609044, 1483.2454568431624, 1502.377311593686, 1521.258000819768, 1540.0844404855327, 1559.0588621416, 1578.0394188515295, 1596.9215520806429, 1615.9925479998735, 1635.0135581066502, 1653.8237509076696, 1672.8362535130436, 1691.772995793539, 1710.7097762845738])
bunch_time_start_data = centers_data[0] - bunch_offset_mapping - bunch_sigma
bunch_time_end_data = centers_data[-1] - bunch_offset_mapping + bunch_sigma

# Approximate MC time to end of the prompt (2us) buffer window to mimic data
data_time_to_end_of_buffer = 2000 - centers_data[-1]
time_to_prompt_end = data_time_to_end_of_buffer + centers[-1]

# 1/3 spill cut we use to reduce skyshine
MC_third_spill_cut = (centers[25] - bunch_offset_mapping + bunch_sigma)
data_third_spill_cut = (centers_data[25] - bunch_offset_mapping + bunch_sigma)

# Approximate exposure scaling between the skyshine sideband (>2-sigma) to our signal (<1-sigma)
# This is calculated by:
# signal (26 bunches, each with a +/-1-sigma cut, corresponding to 3.5ns * 2 of exposure = 182ns)
# skyshine sideband grabs the 25-inter-bunch regions between the 26 bunches, each "side" of the interval
# grabs (18.936/2 - 3.5*2 = 2.468ns --> x2 x25 = 123.4)
EXPOSURE_SCALE = (182/123.4)       # this scales up our calculated skyshine rate within the sideband
                                   # technically we already have the skyshine rate calculated from the sideband script that is properly scaled
                                   # but we can keep this in here for completeness

ALPHA_SCALING = 0.27               # derived dirt scaling factor from our CC sideband

