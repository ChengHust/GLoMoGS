function [rate,error] = Cal_Real_Function(Originaldata,indX)

N = size(indX,1);
D = size(Originaldata,2)-1;

rate  = zeros(N,1);
error = zeros(N,1);


for i = 1 : N 
    
    ind  = indX(i,:);
    
    %% 数据处理
    data  = Originaldata(:,1:end-1);
    label = Originaldata(:,end);
    scale = size(data,1);


    SelectInd = find(ind==1);
    
    data  = data(:,SelectInd);
    
    % 10折交叉
    nFold    = 10;
    indices = crossvalind('Kfold', scale, nFold);

    Accuracy = 0;
    for nF = 1 : nFold 
    
        % 训练集索引 和 测试集索引
        testIndices  = (indices == nF);
        trainIndices   = ~testIndices;
       
        % 训练集和训练标签
        trainData  = data(trainIndices, :);
        trainLabel = label(trainIndices, :);
    
        % 测试集和测试标签
        testData = data(testIndices, :);
        testLabel = label(testIndices, :);
    
        %% KNN分类
        kNNClassifier = fitcknn(trainData, trainLabel, 'NumNeighbors', 5);
        Prelabel = predict(kNNClassifier,testData);
        Acc = sum(Prelabel==testLabel)/length(testLabel);
        Accuracy = Accuracy + Acc;
    end
    Acc = Accuracy / nFold;
     %% 选择特征的比例 和 分类误差
    rate(i,:)  = sum(ind);
    error(i,:) = 1 - Acc;

end