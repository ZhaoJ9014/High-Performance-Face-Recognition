function score = Compute_Similarity(probe_feat,gallery_feat,plda_model)

pairs_num = size(probe_feat,2);
score = zeros(pairs_num,1);
chunks = [1:1000:pairs_num pairs_num+1];
for k = 1:numel(chunks)-1
        kindex = chunks(k):(chunks(k+1)-1);
        chunk_score = plda_eval(plda_model,probe_feat(:,kindex),gallery_feat(:,kindex));
        score(kindex) = diag(chunk_score);
end

end
