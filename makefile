SHELL=/bin/bash

RANGE=$(shell echo {1..20})


# $(info $$RANGE is [${RANGE}])

.PHONY: 

.SECONDARY: %.RData

.NOTPARALLEL: 

NETS = $(wildcard ./networks/net*/checkpoint.pth.tar)

NET_DIRS = $(NETS:/checkpoint.pth.tar=)

# $(info $$NET_DIRS is [${NET_DIRS}])

# SUBSETS = $(subst $(basename $(wildcard ./imagenet_subsets/*.csv)),./imagenet_subsets,)
SUBSETS = $(subst ./imagenet_subsets/,,$(basename $(wildcard ./imagenet_subsets/*.csv)))
$(info $$SUBSETS is [${SUBSETS}])


BASE_DATA_FRAMES = \
	complexity \
	ev10 \
	graph10 \
	graph50 \
	inc10 \
	inc50 \
	test3gap \
	test4add \
	test4coll \
	test5 \
	test6

DATA_FRAMES = $(addprefix data.frames/,$(addsuffix .RData,$(BASE_DATA_FRAMES)))

all: analyses.html

analyses.html: $(DATA_FRAMES)
	Rscript -e 'library(knitr); knit2html("analyses.Rmd")'

$(addsuffix /train,$(SUBSETS)): imagenet
	python3 recreate_subsets.py ../../imagenet

# cifar10: 
	# python3 make_cifar.py 

# imagenet_datasets: $(SUBSETS)
	# python3 -c "from make_imagenet_datasets import main20; main20()"

# base_models: $(wildcar /*/net*/valid

NN_FILES= valid_flatten.npy \
	valid_targets.csv \
	valid_logits.npy \
	test_flatten.npy \
	test_targets.csv \
	test_logits.npy

NJOBS=6

IMAGENET_NETWORKS=$(addsuffix /checkpoint.pth.tar,$(addprefix networks/net, $(RANGE)))

cifar10/net%/nn.tar:
	mkdir -p $(dir $@)
	if [ ! -f $(subst nn.tar,valid_logits.npy,$@) ]; then cd cifar10; python3 ../train_cifar.py --val_size=0 --out_dir $(notdir $(patsubst %/,%,$(dir $@))); fi
	cd $(dir $@); tar -cvf $(notdir $@) $(NN_FILES)

imagenette/eva%/nn.tar: imagenette/train
	mkdir -p $(dir $@)
	if [ ! -f $(subst nn.tar,valid_logits.npy,$@) ]; then cd $(dir $@); python3 ../../main2.py -V 200 -p 50 -j $(NJOBS) -a resnet18 .. ; fi 
	cd $(dir $@); tar -cvf $(notdir $@) $(NN_FILES)


imagenette/net%/nn.tar: $(IMAGENET_NETWORKS) imagenette/train
	mkdir -p $(dir $@)
	if [ ! -f $(subst nn.tar,valid_logits.npy,$@) ]; then cd $(dir $@); python3 ../../main2.py -p 50 --resume ../../networks/$(notdir $(patsubst %/,%,$(dir $@)))/checkpoint.pth.tar --activations -j $(NJOBS) -a resnet18 ..  ; fi 
	cd $(dir $@); tar -cvf $(notdir $@) $(NN_FILES)
	
imagenet-50/net%/nn.tar: $(IMAGENET_NETWORKS) imagenet-50/train
	mkdir -p $(dir $@)
	if [ ! -f $(subst nn.tar,valid_logits.npy,$@) ]; then cd $(dir $@); python3 ../../main2.py -p 50 --resume ../../networks/$(notdir $(patsubst %/,%,$(dir $@)))/checkpoint.pth.tar --activations -j $(NJOBS) -a resnet18 ..  ; fi 
	cd $(dir $@); tar -cvf $(notdir $@) $(NN_FILES)

imagenet-50/eva%/nn.tar: $(IMAGENET_NETWORKS) imagenet-50/train
	mkdir -p $(dir $@)
	if [ ! -f $(subst nn.tar,valid_logits.npy,$@) ]; then cd $(dir $@); python3 ../../main2.py -V 200 -p 50 -j $(NJOBS) -a resnet18 ..  ; fi 
	cd $(dir $@); tar -cvf $(notdir $@) $(NN_FILES)
	
imagewoof/net%/nn.tar: $(IMAGENET_NETWORKS) imagewoof/train
	
imagewoof/net%/nn.tar: $(IMAGENET_NETWORKS) imagewoof/train
	mkdir -p $(dir $@)
	if [ ! -f $(subst nn.tar,valid_logits.npy,$@) ]; then cd $(dir $@); python3 ../../main2.py -p 50 --resume ../../networks/$(notdir $(patsubst %/,%,$(dir $@)))/checkpoint.pth.tar --activations -j $(NJOBS) -a resnet18 ..  ; fi 
	cd $(dir $@); tar -cvf $(notdir $@) $(NN_FILES)


networks/net%/checkpoint.pth.tar:
	mkdir -p $(dir $@)
	cd $(dir $@); python3 ../../main2.py -p 50 -j $(NJOBS) -a resnet18 ../../imagenet 

%valid_flatten.npy: %nn.tar
	cd $(dir $@); tar --skip-old-files -x -v -f  nn.tar $(notdir $@)

%valid_targets.csv: %nn.tar
	cd $(dir $@); tar --skip-old-files -x -v -f  nn.tar $(notdir $@)

%valid_logits.npy: %nn.tar
	cd $(dir $@); tar --skip-old-files -x -v -f  nn.tar $(notdir $@)

%test_flatten.npy: %nn.tar
	cd $(dir $@); tar --skip-old-files -x -v -f  nn.tar $(notdir $@)

%test_targets.csv: %nn.tar
	cd $(dir $@); tar --skip-old-files -x -v -f  nn.tar $(notdir $@)

%test_logits.npy: %nn.tar
	cd $(dir $@); tar --skip-old-files -x -v -f  nn.tar $(notdir $@)

%svm-linear4-1.RData: %valid_flatten.npy %valid_targets.csv
	Rscript -e "source('base.R'); fit.svm('$(dir $<)', kernel = 'linear')"

%platt-linear4-1.RData: %svm-linear4-1.RData 
	Rscript -e "source('pair.R'); prep_platt('$(dir $<)', file = 'svm-linear4-1')"

%svm-radial4-1.RData: %valid_flatten.npy %valid_targets.csv
	Rscript -e "source('base.R'); fit.svm('$(dir $<)', kernel = 'radial')"

%platt-radial4-1.RData: %svm-radial4-1.RData 
	Rscript -e "source('pair.R'); prep_platt('$(dir $<)', file = 'svm-radial4-1')"

%graphs.RData: %platt-radial4-1.RData %platt-linear4-1.RData
	Rscript -e "source('graph_wu.R'); graphs_data('$(dir $<)', reps = 50)"

# VALID_FILES= $(shell find . -type f -name 'valid_logits.npy')
# $(info $$VALID_FILES is [${VALID_FILES}])

# LINEAR_PLATT=$(subst valid_logits.npy,platt-linear4-1.RData,$(VALID_FILES))
# RADIAL_PLATT=$(subst valid_logits.npy,platt-radial4-1.RData,$(VALID_FILES))

SUB20=$(filter-out imagenet-50,$(SUBSETS))
SUB20+=cifar10

KERNELS=linear radial
SVM_FILES20=$(foreach kernel,$(KERNELS), \
	$(foreach subset,$(SUB20), \
		$(addprefix $(subset)/net,  \
			$(addsuffix /svm-$(kernel)4-1.RData,$(RANGE)) \
		) \
	) \
)


NET_DIRS20=$(foreach subset,$(SUB20), \
	$(addprefix $(subset)/net,$(RANGE))) \
	$(addprefix imagenette/eva, $(RANGE)) 


NET_DIRS=$(NET_DIRS20) imagenet-50/net1

NET_FILES=$(foreach nnfile,$(NN_FILES), \
	$(addsuffix /$(nnfile),$(NET_DIRS))) 

NET_FILES_TAR=$(NET_FILES) $(addsuffix /nn.tar,$(NET_DIRS))

# $(info $$NET_FILES_TAR is [${NET_FILES_TAR}])

NET_DIR=NET_DIR20=imagenet-50/net1

EVA_SVM=$(addsuffix /svm-linear4-1.RData, $(addprefix imagenette/eva, $(RANGE)))
EVA_PLATT=$(subst svm,platt,$(EVA_SVM))

SVM_FILES=$(SVM_FILES20) \
	imagenet-50/net1/svm-linear4-1.RData \
	imagenet-50/net1/svm-radial4-1.RData 
# imagenet-50/eva1/svm-linear4-1.RData \
	$(EVA_SVM)

PLATT_FILES=$(subst svm,platt,$(SVM_FILES))

# NOT_GRAPH=$(addsuffix /valid_logits.npy,$(addprefix ./imagenette/eva,$(RANGE))) \
		#./imagenet-50/eva1/valid_logits.npy

GRAPH_FILES_R=$(subst svm-radial4-1.RData,graphs.RData,$(filter-out %linear4-1.RData,$(SVM_FILES)))
GRAPH_FILES=$(GRAPH_FILES_R) $(subst svm-radial4-1.RData,svm-linear4-1.RData,$(GRAPH_R_FILES))
# $(info $$GRAPH_FILES is [${GRAPH_FILES}])

.SECONDARY: $(SVM_FILES) $(PLATT_FILES) $(GRAPH_FILES) $(EVA_PLATT) $(EVA_SVM)

.NOTINTERMEDIATE: $(SVM_FILES) $(PLATT_FILES) $(GRAPH_FILES) $(EVA_PLATT) $(EVA_SVM) $(IMAGENET_NETWORKS) $(NET_FILES)

.INTERMEDIATE: $(addsuffix /nn.tar,$(NET_DIRS))
		
data.frames/test3gap.RData: cifar10 imagenet $(PLATT_FILES) $(SVM_FILES)
	Rscript -e "source('test3gap.R'); df.test3gap <- datasets_df(test_fn = test3gap); save(df.test3gap, file = 'data.frames/test3gap.RData')"

data.frames/test5.RData: cifar10 imagenet $(PLATT_FILES) $(SVM_FILES)
	Rscript -e "source('test5.R'); df.test5 <- datasets_df(test_fn = test5); save(df.test5, file = 'data.frames/test5.RData')"

data.frames/test6.RData: cifar10 imagenet $(PLATT_FILES) $(SVM_FILES)
	Rscript -e "source('test6.R'); df.test6 <- datasets_df(test_fn = test6); save(df.test6, file = 'data.frames/test6.RData')"

data.frames/test4coll.RData: cifar10 imagenet $(PLATT_FILES) $(SVM_FILES)
	Rscript -e "source('test4coll.R'); df.test4coll <- datasets_df(test_fn = test4coll); save(df.test4coll, file = 'data.frames/test4coll.RData')"

data.frames/test4add.RData: cifar10 imagenet $(PLATT_FILES) $(SVM_FILES)
	Rscript -e "source('test4add.R'); df.test4add <- datasets_df(test_fn = test4add); save(df.test4add, file = 'data.frames/test4add.RData')"

data.frames/inc50.RData: cifar10 imagenet $(PLATT_FILES) $(SVM_FILES)
	Rscript -e "source('inc2.R'); df.inc50 <- inc50(); save(df.inc50, file = 'data.frames/inc50.RData')"

data.frames/ev10.RData: cifar10 imagenet $(EVA_SVM) $(EVA_PLATT)
	Rscript -e "source('inc2.R'); df.ev10 <- inc_eva(); save(df.ev10, file = 'data.frames/ev10.RData')"

# imagenet-50/eva1/valid_flatten.npy: 
	# python3 -c "from make_imagenet_datasets import main50; main50()"

# data.frames/ev50.RData: imagenet-50/eva1/platt-linear4-1.RData
	# Rscript -e "source('inc2.R'); df.ev50 <- inc3('imagenet-50/eva1', runs = 50); save(df.ev50, file = 'data.frames/ev50.RData')"

data.frames/inc10.RData: cifar10 imagenet $(PLATT_FILES) $(SVM_FILES)
	Rscript -e "source('inc2.R'); df.inc10 <- inc10(); save(df.inc10, file = 'data.frames/inc10.RData')"

data.frames/graph10.RData: $(GRAPH_FILES) $(PLATT_FILES)
	Rscript -e "source('graph_wu.R'); graphs10()" 

data.frames/graph50.RData: $(GRAPH_FILES) $(PLATT_FILES)
	Rscript -e "source('graph_wu.R'); df.graph <- graphs_data(dir = 'imagenet-50/net1'); save(df.graph, file = 'data.frames/graph50.RData') " 

data.frames/complexity.RData: imagenet-50/net1/platt-radial4-1.RData
	Rscript -e "source('complexity.R'); eval.complexity(time_limit_hr = 12, K = 49)"

imagenet-50/extra_valid/valid_from_train_targets.csv: imagenet
	cd imagenet-50/extra_valid; python ../../main2.py -j 6 -a resnet18 -V 200 ../../imagenet-50
	Rscript -e "source('pair.R'); prep_platt('$(dir $<)', file = 'svm-linear4-1')"


# .ONESHELL:

clean: 
	read -p "Are you sure (CTRL+C if you aren't)? " -n 1 -r  
	find . -type f -name 'svm*' -delete    
	find . -type f -name 'platt*' -delete    
	find . -type f -name '*tar' -delete
	find . -type f -name '*RData' -delete
	find . -type f -name '*npy' -delete

	

