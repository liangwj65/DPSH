function DataPrepare
X = [] ;
L = [] ;
for i=1:5
    clear data labels batch_label;
    load(['./data/cifar-10-batches-mat/data_batch_' num2str(i) '.mat']);
    data = reshape(data',[32,32,3,10000]);
    data = permute(data,[2,1,3,4]);
    X = cat(4,X,data) ;
    L = cat(1,L,labels) ;
end
clear data labels;
load('./data/cifar-10-batches-mat/test_batch.mat');
data=reshape(data',[32,32,3,10000]);
data = permute(data,[2,1,3,4]);
X = cat(4,X,data) ;
L = cat(1,L,labels) ;


test_data = [];
test_L = [];
data_set = [];
dataset_L = [];
train_data = [];
train_L = [];
for label=0:9
    index = find(L==label);
    N = size(index,1) ;
    perm = randperm(N) ;
    index = index(perm);
    
    query_idx = index(1:1000);
    test_data = cat(4, test_data, X(:,:,:,query_idx));
    test_L = cat(1, test_L, L(query_idx));
    
    % 检索数据库：剩余图像
    database_idx = index(1001:end);
    data_set = cat(4, data_set, X(:,:,:,database_idx));
    dataset_L = cat(1, dataset_L, L(database_idx));
    
    % 未标记的训练集：从检索数据库中随机选择 500 张图像
    perm_db = randperm(numel(database_idx));
    train_idx = database_idx(perm_db(1:500));
    train_data = cat(4, train_data, X(:,:,:,train_idx));
    train_L = cat(1, train_L, L(train_idx));
   
end
save('cifar-10.mat','test_data','test_L','data_set','dataset_L','train_data','train_L');
end

