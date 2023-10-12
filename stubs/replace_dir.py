f = open('make_fake_eccodes_def.sh')
lines = f.readlines()
f.close()
for l in lines:
    tmp = l.replace("/gpfsm/dswdev/jcsda/spack-stack/spack-stack-1.4.0/envs/unified-env-v2/install/gcc/10.1.0/eccodes-2.27.0-dprw2fd","${ECCODESDIR}")
    print(tmp)
