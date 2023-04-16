The required packages are listed in environment.yml. Import using

 	> conda env create --name envname --file=environment.yml

To run the simulations, run

	> snakemake --cores N 

where N is the number of cores.