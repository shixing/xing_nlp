CARMEL=./carmel
# EM training
$CARMEL --train-cascade -HJ tagging.data tagging.fsa tagging.fst
# remove *e*
$CARMEL --project-right --project-identity-fsa -HJ tagging.fsa.trained > tagging.fsa.trained.noe
awk 'NF>0' tagging.data > tagging.data.noe

head -n 1 tagging.data.noe | $CARMEL -qbsriWIEk 1 tagging.fsa.trained.noe tagging.fst.trained

echo '"I" "like" "you" "."' | $CARMEL -qbsriWIEk 1 tagging.fsa.trained.noe tagging.fst.trained
