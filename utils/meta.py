# data_root_dir = '/Volumes/data/lupe'
data_root_dir = '/Users/justinjames/LUPE_Corder-Lab'

###### DO NOT CHANGE ANYTHING BELOW ######
behavior_names = ['still',
                  'walking',
                  'rearing',
                  'grooming',
                  'licking hindpaw L',
                  'licking hindpaw R']

behavior_colors = ['crimson',
                   'darkcyan',
                   'goldenrod',
                   'royalblue',
                   'rebeccapurple',
                   'mediumorchid']

keypoints = ["nose", "mouth", "l_forepaw", "l_forepaw_digit", "r_forepaw", "r_forepaw_digit",
             "l_hindpaw", "l_hindpaw_digit1", "l_hindpaw_digit2", "l_hindpaw_digit3",
             "l_hindpaw_digit4", "l_hindpaw_digit5", "r_hindpaw", "r_hindpaw_digit1",
             "r_hindpaw_digit2", "r_hindpaw_digit3", "r_hindpaw_digit4", "r_hindpaw_digit5",
             "genitalia", "tail_base"]

# 30 pixels = 1 cm
pixel_cm = 0.0330828

groups = [f'Group{i}' for i in range(1, 100)]
conditions = [f'Condition{i}' for i in range(1, 100)]
groups_project_DFNZ_formalin = ['C57', 'KiKO_Palmiter']
conditions_project_DFNZ_formalin = ['Saline', 'Formalin']
groups_5xFAD_Capsaicin_3mo_PosNeg = ['LUPE_ ADppp1r3b_LHPcapsaicin_3mo_PosNeg']
conditions_5xFAD_Capsaicin_3mo_PosNeg = ['negative', 'positive']
groups_5xFAD_Capsaicin_4mo_PosNeg = ['LUPE_ ADppp1r3b_LHPcapsaicin_4mo_PosNeg']
conditions_5xFAD_Capsaicin_4mo_PosNeg = ['negative', 'positive']
groups_Analysis_PsiloDoseResponse_122025_Male = ['Male']
conditions_Analysis_PsiloDoseResponse_122025_Male = ['psilo_BL1', '3wkd1', '3wkd2_Drug', '3wkd3_PostDrug', '4wk']
groups_Analysis_PsiloDoseResponse_122025_Female = ['Female']
conditions_Analysis_PsiloDoseResponse_122025_Female = ['psilo_BL1', '3wkd1', '3wkd2_Drug', '3wkd3_PostDrug', '4wk']
groups_Analysis_PsiloDoseResponse_122025_Male_3w1B1 = ['Male']
conditions_Analysis_PsiloDoseResponse_122025_Male_3w1B1 = ['3wk1']
groups_project_13wk1_PsiloMales = ['PsiloMales']
conditions_project_13wk1_PsiloMales = ['Combined']
groups_project_13wk2psilofemales = ['PsiloFemales']
conditions_project_13wk2psilofemales = ['Combined']
groups_project_Wisc_OpioidTGFSaline = ['Formalin_Combined']
conditions_project_Wisc_OpioidTGFSaline = ['SalineVeh', 'TGF']
groups_Analysis_PsiloDoseResponse_8wk = ['Combined']
conditions_Analysis_PsiloDoseResponse_8wk = ['Male', 'Female']
