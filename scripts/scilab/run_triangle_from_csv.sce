repo = pwd();
indir = repo + "/tmp/parity/scilab_inputs";
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
T = csvRead(indir + "/triangle_T.csv");
X = csvRead(indir + "/triangle_X.csv");
G = csvRead(indir + "/triangle_G.csv");
C = csvRead(indir + "/triangle_C.csv");
P = csvRead(indir + "/triangle_P.csv");
dof = 2;

[K,p,q] = init(rows(X),dof);
K = kt3e(K,T,X,G);
p = setload(p,P);
u = solve_lag(K,p,C,dof);
[q,S,E] = qt3e(q,T,X,G,u);
R = reaction(q,C,dof);

csvWrite(u, outdir + "/triangle_u.csv", ",");
csvWrite(q, outdir + "/triangle_q.csv", ",");
csvWrite(S, outdir + "/triangle_S.csv", ",");
csvWrite(E, outdir + "/triangle_E.csv", ",");
csvWrite(R, outdir + "/triangle_R.csv", ",");
csvWrite(X, outdir + "/triangle_X.csv", ",");
csvWrite(T, outdir + "/triangle_T.csv", ",");
csvWrite(C, outdir + "/triangle_C.csv", ",");
csvWrite(P, outdir + "/triangle_P.csv", ",");
exit;
