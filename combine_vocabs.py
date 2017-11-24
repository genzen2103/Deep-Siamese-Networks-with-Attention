
vocab1 = { key:int(value) for key,value in [ row.split(":") for row in open('../Vocabs/Quora_vocab.txt','r')] }

vocab2 = { key:int(value) for key,value in [ row.split(":") for row in open('../Vocabs/Yahoo_Webscope_L5_vocab.txt','r')] }

vocab3 = { key:int(value) for key,value in [ row.split(":") for row in open('../Vocabs/Zhang_yahoo_vocab.txt','r')] }

complete_vocab = sorted( set ( vocab1.keys() + vocab2.keys() + vocab3.keys() ) )


file=open("../Vocabs/combined_vocab.txt",'w')
index=0
for k in complete_vocab:
	file.write(k+":"+str(index)+"\n")
	index+=1
file.close()
print ("Vocab Length:",len(complete_vocab))

