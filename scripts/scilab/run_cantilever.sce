repo = pwd();
outdir = repo + "/tmp/parity/scilab_outputs";
if ~isdir(repo + "/tmp") then
    mkdir(repo + "/tmp");
end
if ~isdir(repo + "/tmp/parity") then
    mkdir(repo + "/tmp/parity");
end
if ~isdir(outdir) then
    mkdir(outdir);
end

getd(repo + "/macros");
exec(repo + "/examples/canti.sce", -1);

[K,p,q] = init(size(X,1),dof);
K = kq4e(K,T,X,G);
p = setload(p,P);
[K,p] = setbc(K,p,C,dof);
u = inv(K)*p;
[q,S,E] = qq4e(q,T,X,G,u);
R = reaction(q,C,dof);

csvWrite(u, outdir + "/cantilever_u.csv", ",");
csvWrite(q, outdir + "/cantilever_q.csv", ",");
csvWrite(S, outdir + "/cantilever_S.csv", ",");
csvWrite(E, outdir + "/cantilever_E.csv", ",");
csvWrite(R, outdir + "/cantilever_R.csv", ",");
csvWrite(X, outdir + "/cantilever_X.csv", ",");
csvWrite(T, outdir + "/cantilever_T.csv", ",");
csvWrite(C, outdir + "/cantilever_C.csv", ",");
csvWrite(P, outdir + "/cantilever_P.csv", ",");
exit;
