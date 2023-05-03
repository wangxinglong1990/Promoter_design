# Promoter_design
Please cite us at: PromoR and PromoS for E. coli promoter recognition and classification. https://doi.org/10.1101/2023.03.05.531155
## Aim
design E. coli constitutive promoters
## Design Method:
### DRSAdesign
Promoters are generated using diffusion model (PromoDiff), and filtered functional promoters using protein recognition model (PromoR).\
Promoters can be predicted their strong or weak using PromoS, and their transcription level using PromoA.\
Please keep "Diff.py", "Diff_modules.py", "Diff_model.pth", "PA_modules.py", "PA_best_model.pth", "PS_modules.py", "PS_best_model.pth", "PR_modules.py", "PR_best.pth" in the run folder. These files supplemented in "PromoDiff_generate_promoters\app"\
Sample command for creating 100000 random sequences and predict their strength: python Diff.py -predict yes -device cpu -number_of_created_promoter 100000\
About output: First column, 1 indicates real, 0 indicates false; Second column, 1 indicates strong, 0 indicates weak; Third column, values indicated transcription level\
For only generating sequences and no prediction needed, please repalce "-predict yes" by "-predict no"

### Ndesign
Promoters were generated using constraint-based method, by maintaining the -35 and -10 with a sequence of TTGAGC and TATAAT, and the spacer length ranges from 16 to 18 bp. The other regions are filled with random nucleotides.\
Generated random promoters are predicted their strength using PromoNet.\
Please keep "PN.py", "PN_modules.py", "PN_best_model.pth", "PA_modules.py", "PA_best_model.pth", "PS_modules.py", "PS_best_model.pth" in the run folder. These files supplemented in "PromoNet_strength_prediction\app"\
Sample command for creating 100000 random sequences: python PN.py -sample_type generate_sample -number_of_created_promoter 100000\
About output: First column, 1 indicates strong, 0 indicates weak; Second column, values indicated transcription level; Third column, values indicated promoter strength\
I recommand only focus on the third column

### Random sequences assembling to functional promoters
Promoters were generated randomly, and filtered functional promoters using PromoR, PromoS, and PromoA.\
Please keep "PSA.py", "PR_modules.py", "PR_best.pth", "PA_modules.py", "PA_best_model.pth", "PS_modules.py", "PS_best.pth" in the run folder. These files supplemented in "PromoS_and_PromoA_strength_prediction\app"\
Sample command: python PSA.py -sample_type input_sample -input_file sample.txt -device "cpu"\
About output: First column, 1 indicates real, 0 indicates false; Second column, 1 indicates strong, 0 indicates weak; Third column, values indicated transcription level

## Prediction Function
### Predicting real or false (PromoR) promoter based on given 50 bp sequence

Please keep "PR.py", "PR_modules.py", "PR_best.pth", "sample.txt" in the run folder. These files supplemented in "PromoR_promoter_real_fake_prediction\app"\
Sample command: python PR.py -load_file sample.txt -select_device "cpu"\
About output: 1 indicates real, 0 indicates false\
Note: "sample.txt" prepared as file.

### Predicting real or false (PromoR), strong or weak (PromoS), and transcription level (PromoA) based on given 50 bp sequence
Please keep "PSA.py", "PR_modules.py", "PR_best.pth", "PA_modules.py", "PA_best.pth", "PS_modules.py", "PS_best.pth","sample.txt" in the run folder. These files supplemented in "PromoS_and_PromoA_strength_prediction\app"\
Sample command: python PSA.py -sample_type input_sample -input_file sample.txt -device "cpu"\
About output: First column, 1 indicates real, 0 indicates false; Second column, 1 indicates strong, 0 indicates weak; Third column, values indicated transcription level\
Note: "sample.txt" prepared as above.

### Predicting constraint-based promoter activity with PromoNet

Promoters must be maintaining the -35 and -10 with a sequence of TTGAGC and TATAAT, and the spacer length ranges from 16 to 18 bp.\
Please keep "PN.py", "PN_modules.py", "PN_best_model.pth", "PA_modules.py", "PA_best_model.pth", "PS_modules.py", "PS_best_model.pth","sample.txt" in the run folder. These files supplemented in "PromoNet_strength_prediction\app"\
Sample command: python PN.py -sample_type input_sample -input_file sample.txt -device "cpu"\
About output: First column, 1 indicates strong, 0 indicates weak; Second column, values indicated transcription level; Third column, values indicated promoter strength\
Note: "sample.txt" prepared as above.

## Retrain the network
### PromoS
Navigate to "PromoS_and_PromoA_strength_prediction\PromoS_train", command "python train.py"
### PromoA
Navigate to "PromoS_and_PromoA_strength_prediction\PromoA_train", command "python train.py"
### PromoR
Navigate to "PromoR_promoter_real_fake_prediction\train", command "python train.py"
### PromoDiff
Navigate to "PromoDiff_generate_promoters\train", command "python train.py"
### PromoNet
Navigate to "PromoNet_strength_prediction\train", command "python train.py"

## Datasets
When using datasets, please cite:
RegulonDB(1): Gama-Castro, S., Salgado, H., Santos-Zavaleta, A., Ledezma-Tejeida, D., Muñiz-Rascado, L., García-Sotelo, J.S., Alquicira-Hernández, K., Martínez-Flores, I., Pannier, L., Castro-Mondragón, J.A. et al. (2015) RegulonDB version 9.0: high-level integration of gene regulation, coexpression, motif clustering and beyond. Nucleic Acids Res, 44, D133-D143.\
E. coli nature promoter(2): Thomason, M.K., Bischler, T., Eisenbart, S.K., Förstner, K.U., Zhang, A., Herbig, A., Nieselt, K.K., Sharma, C.M. and Storz, G. (2014) Global Transcriptional Start Site Mapping Using Differential RNA Sequencing Reveals Novel Antisense RNAs in Escherichia coli. J. Bacteriol, 197, 18 - 28.\
E. coli genome(3): Jeong, H., Barbe, V., Lee, C.H., Vallenet, D., Yu, D.S., Choi, S.-H., Couloux, A., Lee, S.-w., Yoon, S.H., Cattolico, L. et al. (2009) Genome sequences of Escherichia coli B strains REL606 and BL21(DE3). J. Mol. Biol, 394 4, 644-652.

## Environments
Operating System: Linux or Windows.\
Python 3.6 or newer installation.\
One can simply use environment.yml to install all the dependencies:\
Command: conda env create --file environment.yml

--------------------------------------------------------------------------
Acknowledgement:
Kindly thanks for Professor Jingwen Zhou for providing support for my ideology and experiments.
Diffusion model was inspired by GitHub author "dome272" of the repository "Diffusion-Models-pytorch". Below is the link.
https://github.com/dome272/Diffusion-Models-pytorch

Auther description:
Quit interested in protein engineering and metabolic engineering.\
Good at protein expression using E. coli platform, protein purification, and function characterization.\
Specific in protein function mechanisms.\
Starting wet experiment by 2015 in La Trobe university, supervised by Marc Kvansakul\
Done PhD in Jiangnan university, Supervised by Jian Chen\
Doing PosDoc in Jiangnan university, Supervised by Jingwen Zhou.
