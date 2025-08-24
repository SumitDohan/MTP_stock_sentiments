# ----- deploy_local.ps1 -----
$localImageName   = "sumitdohan/finbert-mlops"
$localTag         = "latest"
$registryUrl      = "172.26.1.219"
$registryImage    = "pensive_jemison/finbert-mlops"
$registryUsername = "pensive_jemison"
$registryPassword = "27swetadey"

docker tag "$localImageName:$localTag" "$registryUrl/$registryImage:$localTag"
docker login $registryUrl --username $registryUsername --password $registryPassword
docker push "$registryUrl/$registryImage:$localTag"
