import sys
import math
import os

def ngrams(sentence,n):
	n_gms = []
	if n ==1:
		n_gms = sentence
	if n == 2:
		for i in range(len(sentence)-1):
			n_gms.append((sentence[i],sentence[i+1]))
	if n == 3:
		for i in range(len(sentence)-2):
			n_gms.append((sentence[i],sentence[i+1],sentence[i+2]))
	if n == 4:
		for i in range(len(sentence)-3):
			n_gms.append((sentence[i],sentence[i+1],sentence[i+2],sentence[i+3]))
	return n_gms

def count_occurences(data):
	occur = {}
	for i in data:
		if i in occur:
			occur[i] += 1
		else:
			occur[i] = 1
	return occur

def modified_precision(candidates,all_references,n):
	count_clip = 0
	count_w = 0
	for candidate,references in zip(candidates,all_references):
		refined_counts = {}
		candidate = candidate.split(" ")
		n_gram_cand = ngrams(candidate,n)
		if len(n_gram_cand) == 0:
			continue
		cand_count = count_occurences(n_gram_cand)
			
		max_count = {}
		for reference in references:
			reference = reference.replace("\n","")
			reference = reference.split(" ")
			n_gram_ref = ngrams(reference,n)
			ref_count = count_occurences(n_gram_ref)

			for i in cand_count:
				try:
					max_count[i] = max(max_count.get(i,0),ref_count[i])
				except:
					max_count[i] = max_count.get(i,0) 
			
		for i in cand_count:
			refined_counts[i] = min(cand_count[i],max_count[i])

		count_clip += sum(refined_counts.values())
		count_w += sum(cand_count.values())

	m_precision = float(count_clip)/count_w 
	return m_precision

def brevity_penalty(candidate,references):   # bp较短句惩罚
	c = 0
	
	for i in candidate:
		c += len(i)

	find_min = []
	for i in references:
		r = 0
		for j in i:
			r += len(j)
		find_min.append(abs(r-c))

	r = min(find_min)

	if c > r:
		return 1
	else:
		return math.exp(1-r/c)

def bleu_score(candidate,references):
	weights = [0.25,0.25,0.25,0.25]

	pn = []
	bp = brevity_penalty(candidate,references)

	
	data = []
	for i in range(len(references[0])):  # references里面有多少个句子
		temp = []
		for j in range(len(references)):
			#print [references[j][i]]
			temp += [references[j][i]]
		data.append(temp)


	for i in range(1,5):   # 1-4grams
		temp = modified_precision(candidate,data,i)
		#print temp
		pn.append(temp)



	p_sum = 0
	for w,p in zip(weights,pn):
		if p!=0:
			p_sum += w*math.log(p)

	
	score = bp*math.exp(p_sum)

	return score, pn


def BLEU(arg1, arg2):
    candidate_file = arg1
    reference_file  = arg2

    f = open(candidate_file,'r')
    candidate_contents = f.readlines()

    reference_contents = []
    paths = []

    if os.path.isfile(reference_file):
        print('Yes!It\'s a file')
        f1 = open(reference_file,'r')
        reference_contents.append(f1.readlines())
    else:
        for dirpath,dirname,files in os.walk(reference_file):
            for filename in files:
                paths.append(os.path.join(dirpath, filename))

        for path in paths:
            f1 = open(path,'r')
            reference_contents.append(f1.readlines())
            f1.close()


    for i in range(len(candidate_contents)):
        candidate_contents[i] = candidate_contents[i].replace("\n","")

    score, pn = bleu_score(candidate_contents,reference_contents)
    f.close()
    return score, pn

