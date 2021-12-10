from func import *
import time
txt_path = f"{args.graph_dir}/sentiment_data_4.txt"
csv_path = 'data/sentiment_data_15000_4.csv'
adj,features,nfeat_dim = Prepare_Sentiment_txt(txt_path )
nclass,labels,indexs,label2id = Prepare_Sentiment_csv(csv_path)

model = GCN(nfeat=nfeat_dim,nhid=2048,nclass=nclass)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
model = model.to(args.device)
start = time.time()
train(model,optimizer,indexs,labels,features, adj,nclass,label2id,epochs=10)
end = time.time()
print('run time:',end-start)
