# This program is used to calculate the 
# spectral similarity of 2 songs
# where the input should be a wav file



import numpy as np
from collections import Counter
from pylab import *
import warnings
from scipy.io import wavfile as wv
from scipy.cluster.vq import kmeans,vq
import python_speech_features as sf
from pylab import *
warnings.filterwarnings("ignore")


# File 1 and File 2 contains the path of the wav files
# Change file1 and flie2 to check similarity on other songs

file1 = "./song1.wav"
file2 = "./song2.wav"
lifter = 0

# numcep for a music file is taken to be 12
numcep = 12	
v=3
code=[]

sb=[]

def feat(wav,c=False,code=[],lifter=0,numcep=12,v=3):
	fs,s=wv.read(wav)
	mf=sf.mfcc(s,samplerate=fs,numcep=numcep,ceplifter=lifter)
	norm_feat=[]
	for i,feat in enumerate(mf):
		der = np.gradient(feat)
		der2 = np.gradient(feat,2)
		der = np.concatenate((feat,der,der2))
		norm_feat.append((der-np.mean(der))/np.std(der))
	if c==True:
		codebook, distortion = kmeans(norm_feat, v)
	else:
		codebook = code
	codewords, dist = vq(norm_feat, codebook)
	sb.append(codewords)
	histo = np.array(list(Counter(codewords).values()))#/len(mf)
	print(wav,"\t",histo)
	return histo,codebook,sb

# This function may or may not be used
# This is just to display the histogram

def plot_(val,data):
	s=subplot(val)
	title("Histogram-"+str(val)[2])
	s.set_xlabel("Code")
	s.set_ylabel("Frequency")
	s.legend()
	hist(data)
a,code,sbp = feat(file1,True,v=v)
b,code,sbp = feat(file2,code=code,v=v)
figure(1)
plot_(211,sb[0])
plot_(212,sb[1])
show()

# Lower the value of ans1 higher is the similarity
ans1 = np.linalg.norm(a-b)
print(ans1)