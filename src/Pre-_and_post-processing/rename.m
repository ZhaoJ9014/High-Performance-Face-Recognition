clc
clear all;

fileFolder=fullfile('/home/facemanLV/Documents/DataSets/buffer');
dirOutput=dir(fullfile(fileFolder));
fileNames={dirOutput.name}';

for i = 3:length(fileNames)
    disp('====');
    disp(i);
    disp('====');
    disp(length(fileNames)-2);
    disp('====');
    source = strcat('/home/facemanLV/Documents/DataSets/buffer/', fileNames{i,1});
    [a, b, ext] = fileparts(fileNames{i,1});
    destination = strcat('/home/facemanLV/Documents/DataSets/ABC_', num2str(i-2), ext);
    copyfile(source,destination);
end
