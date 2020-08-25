N=10
base="."

# Simulate

pipenv run python -u simulate.py --fixz --fixm --fixalign -n $N --name train_fix --dir $base
pipenv run python -u simulate.py --fixz --fixalign -n $N --name train_mass --dir $base
pipenv run python -u simulate.py --fixz --fixm -n $N --name train_align --dir $base
pipenv run python -u simulate.py -n $N --name train_full --dir $base

pipenv run python -u simulate.py --calibrate --theta 1 --fixz --fixm --fixalign -n $N --name calibrate_fix_theta --dir $base
pipenv run python -u simulate.py --calibrate --theta 1 --fixz --fixalign -n $N --name calibrate_mass_theta --dir $base
pipenv run python -u simulate.py --calibrate --theta 1 --fixz --fixm -n $N --name calibrate_align_theta --dir $base
pipenv run python -u simulate.py --calibrate --theta 1 -n $N --name calibrate_full_theta --dir $base

pipenv run python -u simulate.py --calref --fixz --fixm --fixalign -n $N --name calibrate_fix_ref --dir $base
pipenv run python -u simulate.py --calref --fixz --fixalign -n $N --name calibrate_mass_ref --dir $base
pipenv run python -u simulate.py --calref --fixz --fixm -n $N --name calibrate_align_ref --dir $base
pipenv run python -u simulate.py --calref -n $N --name calibrate_full_ref --dir $base

pipenv run python -u simulate.py --fixz --fixm --fixalign -n $N --name test_fix --test --point --dir $base
pipenv run python -u simulate.py --fixz --fixalign -n $N --name test_mass --test --point --dir $base
pipenv run python -u simulate.py --fixz --fixm -n $N --name test_align --test --point --dir $base
pipenv run python -u simulate.py -n $N --name test_full --test --point --dir $base

pipenv run python -u simulate.py --fixz --fixm --fixalign -n $N --name test_fix_prior --test --dir $base
pipenv run python -u simulate.py --fixz --fixalign -n $N --name test_mass_prior --test --dir $base
pipenv run python -u simulate.py --fixz --fixm -n $N --name test_align_prior --test --dir $base
pipenv run python -u simulate.py -n $N --name test_full_prior --test --dir $base

Combination

pipenv run python -u combine_samples.py --regex train_fix "train_fix" --dir $base
pipenv run python -u combine_samples.py --regex test_fix_point "test_fix" --dir $base
pipenv run python -u combine_samples.py --regex test_fix_prior "test_fix_prior" --dir $base

pipenv run python -u combine_samples.py --regex train_mass "train_mass" --dir $base
pipenv run python -u combine_samples.py --regex test_mass_point "test_mass" --dir $base
pipenv run python -u combine_samples.py --regex test_mass_prior "test_mass_prior" --dir $base

pipenv run python -u combine_samples.py --regex train_align "train_align" --dir $base
pipenv run python -u combine_samples.py --regex test_align_point "test_align" --dir $base
pipenv run python -u combine_samples.py --regex test_align_prior "test_align_prior" --dir $base

pipenv run python -u combine_samples.py --regex train_full "train_full" --dir $base
pipenv run python -u combine_samples.py --regex test_full_point "test_full" --dir $base
pipenv run python -u combine_samples.py --regex test_full_prior "test_full_prior" --dir $base

# Training

pipenv run python -u train.py carl train_fix carl_fix --dir $base --epochs 5
pipenv run python -u train.py carl train_mass carl_mass --load carl_fix --dir $base --epochs 5
pipenv run python -u train.py carl train_align carl_align --load carl_fix --dir $base --epochs 5
pipenv run python -u train.py carl train_full carl_full --load carl_fix --dir $base --epochs 5

pipenv run python -u train.py alices train_fix alices_fix --dir $base --epochs 5
pipenv run python -u train.py alices train_mass alices_mass --load alices_fix --dir $base --epochs 5
pipenv run python -u train.py alices train_align alices_align --load alices_fix --dir $base --epochs 5
pipenv run python -u train.py alices train_full alices_full --load alices_fix --dir $base --epochs 5

pipenv run python -u train.py alices train_full alices_full_alpha2e2 --load alices_fix --alpha 2.e-2 --dir $base --epochs 5
pipenv run python -u train.py alices train_full alices_full_alpha2e3 --load alices_fix --alpha 2.e-3 --dir $base --epochs 5
pipenv run python -u train.py alices train_full alices_full_alpha2e5 --load alices_fix --alpha 2.e-5 --dir $base --epochs 5
pipenv run python -u train.py alices train_full alices_full_alpha2e6 --load alices_fix --alpha 2.e-6 --dir $base --epochs 5

pipenv run python -u train.py alices train_full alices_full_lr1e3 --load alices_fix --lr 1.e-3 --dir $base --epochs 5
pipenv run python -u train.py alices train_full alices_full_lr3e4  --load alices_fix --lr 3.e-4 --dir $base --epochs 5
pipenv run python -u train.py alices train_full alices_full_lr3e5  --load alices_fix --lr 3.e-5 --dir $base --epochs 5
pipenv run python -u train.py alices train_full alices_full_lr1e5  --load alices_fix --lr 1.e-5 --dir $base --epochs 5

pipenv run python -u train.py alices train_full alices_full_sgd1e1 --load alices_fix --optimizer sgd --lr 0.1 --dir $base --epochs 5
pipenv run python -u train.py alices train_full alices_full_sgd1e2 --load alices_fix --optimizer sgd --lr 0.01 --dir $base --epochs 5

pipenv run python -u train.py alices train_full alices_full_fromscratch --dir $base --epochs 5
pipenv run python -u train.py alices train_full alices_full_deep --load alices_fix --deep --epochs 5 --dir $base --epochs 5
pipenv run python -u train.py alices train_full alices_full_batchsize64 --load alices_fix --batchsize 64 --dir $base --epochs 5
pipenv run python -u train.py alices train_full alices_full_batchsize256 --load alices_fix --batchsize 256 --dir $base --epochs 5

# Evaluation

for tag in fix mass align full
do
    modeltag=${tag}
    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on prior sample"
    echo ""
    pipenv run python -u test.py carl_${modeltag} test_${tag}_prior carl_${modeltag}_prior --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on shuffled prior sample"
    echo ""
    pipenv run python -u test.py carl_${modeltag} test_${tag}_prior carl_${modeltag}_shuffledprior --shuffle --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on point sample / param grid"
    echo ""
    pipenv run python -u test.py carl_${modeltag} test_${tag}_point carl_${modeltag}_grid --grid --dir $base

done

for tag in fix mass align full
do
    modeltag=${tag}
    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on prior sample"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_prior alices_${modeltag}_prior --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on shuffled prior sample"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_prior alices_${modeltag}_shuffledprior --shuffle --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on point sample / param grid"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_point alices_${modeltag}_grid --grid --dir $base

done

tag=full
for variation in alpha2e2 alpha2e3 alpha2e5 alpha2e6
do
    modeltag=${tag}_${variation}
    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on prior sample"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_prior alices_${modeltag}_prior --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on shuffled prior sample"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_prior alices_${modeltag}_shuffledprior --shuffle --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on point sample / param grid"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_point alices_${modeltag}_grid --grid --dir $base

done

tag=full
for variation in lr1e3 lr3e4 lr3e5 lr1e5
do
    modeltag=${tag}_${variation}
    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on prior sample"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_prior alices_${modeltag}_prior --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on shuffled prior sample"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_prior alices_${modeltag}_shuffledprior --shuffle --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on point sample / param grid"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_point alices_${modeltag}_grid --grid --dir $base

done

tag=full
for variation in sgd1e1 sgd1e2
do
    modeltag=${tag}_${variation}
    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on prior sample"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_prior alices_${modeltag}_prior --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on shuffled prior sample"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_prior alices_${modeltag}_shuffledprior --shuffle --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on point sample / param grid"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_point alices_${modeltag}_grid --grid --dir $base

done


tag=full
for variation in batchsize64 fromscratch deep batchsize256
do
    modeltag=${tag}_${variation}
    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on prior sample"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_prior alices_${modeltag}_prior --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on shuffled prior sample"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_prior alices_${modeltag}_shuffledprior --shuffle --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on point sample / param grid"
    echo ""
    pipenv run python -u test.py alices_${modeltag} test_${tag}_point alices_${modeltag}_grid --grid --dir $base

done

# Calibration

pipenv run python -u calibrate.py alices_full_grid alices_calibrate_full --dir $base --bins 20 --name 20bins
pipenv run python -u calibrate.py alices_full_grid alices_calibrate_full --dir $base --bins 40 --name 40bins
pipenv run python -u calibrate.py alices_full_grid alices_calibrate_full --dir $base --bins 60 --name 60bins
pipenv run python -u calibrate.py alices_full_grid alices_calibrate_full --dir $base --bins 20 -s --name 20sbins
pipenv run python -u calibrate.py alices_full_grid alices_calibrate_full --dir $base --bins 40 -s --name 40sbins
pipenv run python -u calibrate.py alices_full_grid alices_calibrate_full --dir $base --bins 60 -s --name 60sbins
