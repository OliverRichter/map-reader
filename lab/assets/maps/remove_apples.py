import glob
for filename in glob.glob('*.map'):
    f = open(filename)
    lines = f.readlines()
    f.close()
    f = open(filename,'w')
    remove = 0
    for i,line in enumerate(lines):
        if len(lines)>i+2 and line=='\n' and lines[i+1]=='{\n' and lines[i+2]=='  "classname" "apple_reward"\n':
            remove = 4
        elif remove>0:
            remove-=1
        else:
            f.write(line)
    f.close()