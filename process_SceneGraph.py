

import json
import time
import sys
import importlib
importlib.reload(sys)
#sys.setdefaultencoding('utf-8')


print("starting to read filtered_region_graphs.txt\n")
start_time = time.time()
with open('/home/heng/python_py/filtered_region_graphs.txt','r',encoding='utf-8') as f:
	lines = f.readlines()
end_time = time.time()
print("finished reading filtered_region_graphs.txt with " + str((end_time - start_time) * 1000) + '\n')

# 将每行的scene graph读出来
scene_graphs =[]
for line in lines:
	scene_graph = eval(line[:-1])
	scene_graph = scene_graph['regions']   # scene_graph此时是一个list，去掉了image_id，list中的每个元素是一个字典（region graph）
	scene_graphs.append(scene_graph)

fwrite = open('/home/heng/python_py/encodingText2.txt','w')
fword = open('/home/heng/python_py/word2.txt','w')
fphrase = open('/home/heng/python_py/phrase2.txt','w')

print("starting to process graphs\n")
start_time = time.time()
# 对每一个scene_graph进行处理
for scene_graph in scene_graphs:        # 每个scene_graph是源文件中的一行
	for region_graph in scene_graph:    # region_graph是字典类型，每一个就是scene_graph的一个小部分
		#text  后面用于记录要写的关系和属性
		relationships = region_graph['relationships']    # list类型
		attributes = region_graph['attributes']          # list类型
		objects = region_graph['objects']				 # list类型,接下来看看objects是否为空
		synsets = region_graph['synsets']
		phrase = region_graph['phrase']
		if len(objects) > 0:
			if len(relationships) == 0:        # relationships为空,则去看attributes
				arr = []
				for attribute in attributes:   # attribute是dict类型
					Flag_attribute = 0
					Flag_object = 0
					#names = attribute['names']    # 拿这个names处理问题很大，应该用object_id
					object_id = attribute['object_id']
					for obj in objects:     # obj是字典类型
						obj_id = obj['object_id']
						if obj_id == object_id:
							Flag_object = 1
							name = obj['name']
							break
							
					if 'attributes' in attribute.keys():  # attribute中有attributes属性的话
						Flag_attribute = 1
						sub_attributes = attribute['attributes']    # sub_attributes 是list，代表每个region下的attributes字典下的每个列表元素的attributes字典
					if Flag_attribute == 0 and Flag_object == 0:    # attribute中没有attributes属性的话
						text = attribute['names'][0]
						arr.append(attribute['names'][0])
						fwrite.write(text+'  ')
					elif Flag_attribute == 0 and Flag_object == 1:  # attributes中有对象而没有子属性
						text = name
						arr.append(name)
						fwrite.write(text+'  ')
					elif Flag_attribute == 1 and Flag_object == 1:  # attributes中有对象和子属性
						for sub_attribute in sub_attributes:        # 每个object的属性的个数
							if sub_attribute in phrase:
								#text = '('+sub_attribute+','+names[0]+')'
								arr.append(sub_attribute)
								arr.append(name)
								text = sub_attribute+','+name
								fwrite.write(text+'  ')
				fwrite.write('\n')
				arr_set = set(arr)
				fword.write(str(arr_set)+'\n')
				
				phrase = phrase.replace(' ','  ').replace('.','').replace('\n','')     # 所有单空格换成双空格，去除末尾的'.'
				for a in arr_set:
					newa = a.replace(' ','  ')
					if newa in phrase:
						phrase = phrase.replace(newa,a)
				fphrase.write(phrase+'\n')      # 需要处理掉最后一个 ‘.’
			
			if len(relationships) == 1:     # 关系的数量等于1，列出两个二元组
				predicate = relationships[0]['predicate'].lower()    # 获取predicate并转为小写
				if len(objects) == 2 or len(objects) == 1:   # objects中有两个对象或者一个对象
					arr = []
					object_id = relationships[0]['object_id']
					subject_id = relationships[0]['subject_id']
					for obj in objects:  # 寻找对应, objects是字典
						obj_id = obj['object_id']
						if len(obj['synsets']) > 0:
							synset_name = obj['synsets'][0]
							if object_id == obj_id:   # 受
								for synset in synsets:
									if synset['synset_name'] == synset_name:
										entity_name = synset['entity_name']
										text = predicate+','+entity_name
										arr.append(predicate)
										arr.append(entity_name)
										fwrite.write(text+'  ')
										break
							elif subject_id == obj_id: # 主
								for synset in synsets:
									if synset['synset_name'] == synset_name:
										entity_name = synset['entity_name']
										arr.append(entity_name)
										arr.append(predicate)
										text = entity_name+','+predicate
										fwrite.write(text+'  ')
										break
					for attribute in attributes:   # attribute是dict类型
						Flag_attribute = 0
						Flag_object = 0
						object_id = attribute['object_id']
						for obj in objects:     # obj是字典类型
							obj_id = obj['object_id']
							if obj_id == object_id:
								Flag_object = 1
								name = obj['name']
								break
						if 'attributes' in attribute.keys():  # attribute中有attributes属性
							Flag_attribute = 1
							sub_attributes = attribute['attributes']
						if Flag_attribute == 0 and Flag_object == 0:    # attribute中没有attributes属性的话
							text = attribute['names'][0]
							arr.append(attribute['names'][0])
							fwrite.write(text+'  ')
						elif Flag_attribute == 0 and Flag_object == 1:  # attributes中有对象而没有子属性
							text = name
							arr.append(name)
							fwrite.write(text+'  ')
						elif Flag_attribute == 1 and Flag_object == 1:  # attributes中有对象和子属性
							for sub_attribute in sub_attributes:
								if sub_attribute in phrase:
									#text = '('+sub_attribute+','+names[0]+')'
									arr.append(sub_attribute)
									arr.append(name)
									text = sub_attribute+','+name
									fwrite.write(text+'  ')
					fwrite.write('\n')
					phrase = region_graph['phrase']
					arr_set = set(arr)
					fword.write(str(arr_set)+'\n')
					phrase = phrase.replace(' ','  ').replace('.','').replace('\n','')     # 所有单空格换成双空格，去除末尾的'.'
					for a in arr_set:
						newa = a.replace(' ','  ')
						if newa in phrase:
							phrase = phrase.replace(newa,a)
					#fphrase.write(phrase.encode('utf-8'))
					#print(phrase)
					fphrase.write(phrase)
					fphrase.write('\n')
			if len(relationships) > 1:
				pass
end_time = time.time()
print("finished processing with " + str((end_time - start_time) * 1000) + '\n')
fwrite.close()
fphrase.close()	
fword.close()

 			
				
