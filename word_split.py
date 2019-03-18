



fwrite = open('/home/heng/python_py/word_split2.txt','w')
with open('/home/heng/python_py/word2.txt','r') as f:
    lines = f.readlines()
for line in lines:
    line = list(eval(line[:-1]))
    for i in range(len(line)-1):
        fwrite.write(line[i]+'  ')
    fwrite.write(line[len(line)-1]+'\n')
fwrite.close()