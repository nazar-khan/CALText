import numpy as np

def cmp_result(label,rec):
    dist_mat = np.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)
    
def convert(list): 
      
    # Converting integer list to string list 
    s = [str(i) for i in list] 
      
    # Join list items using join() 
    res = "".join(s)
      
    return(res) 
    
def process_chr_error(recfile, labelfile, resultfile):
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    rec_mat = {}
    label_mat = {}
    cc=1
    with open(recfile) as f_rec:
        for line in f_rec:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            rec_mat[key] = latex
            cc=cc+1
    cc=1
    with open(labelfile) as f_label:
        for line in f_label:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            label_mat[key] = latex
            cc=cc+1
    for key_rec in rec_mat:
        label = label_mat[key_rec]
        rec = rec_mat[key_rec]
        dist, llen = cmp_result(label, rec)
        total_dist += dist
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec += 1
    chr_error = float(total_dist)/total_label
    return chr_error
    
def process_wer_error(recfile, labelfile, resultfile,space_ind):
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    rec_mat = {}
    label_mat = {}
    cc=1
    with open(recfile) as f_rec:
        for line in f_rec:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            ss=convert(latex)
            latex = ss.split(str(space_ind))
            rec_mat[key] = latex
            cc=cc+1
    cc=1
    with open(labelfile) as f_label:
        for line in f_label:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            ss=convert(latex)
            latex = ss.split(str(space_ind))
            label_mat[key] = latex
            cc=cc+1
    for key_rec in rec_mat:
        label = label_mat[key_rec]
        rec = rec_mat[key_rec]
        dist, llen = cmp_result(label, rec)
        total_dist += dist
        total_label += llen
    wer = float(total_dist)/total_label
    return wer
    
def process(recfile, labelfile, resultfile,space_ind):
		cer=process_chr_error(recfile, labelfile, resultfile)
		wer=process_wer_error(recfile, labelfile, resultfile,space_ind)
		f_result = open(resultfile,'w')
		f_result.write('CER {}\n'.format(cer))
		f_result.write('WER {}\n'.format(wer))
		f_result.close()