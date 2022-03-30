float w[5][5],ew[5][5];
float x[6]={0,2.5,5,7.5,10,200}; //pt
float y[6]={0,50,100,150,200,2000}; //p

int pbin(float  P)
{
	for(int i=0;i<6;++i) {
		if(P>=y[i]&&P<y[i+1]) return i;
	}
	return 5;
}

int ptbin(float  Pt)
{
	for(int i=0;i<5;++i) {
		if(Pt>=x[i]&&Pt<x[i+1]) return i;
	}
	return 5;
}

float weight(float P, float Pt)
{
	float p=P/1000.0;
	float pt=Pt/1000.0;
	int ip = pbin(p);
	int ipt = ptbin(pt);
	return w[ipt][ip];
}

void addpptw(const char* input, const char* output, const char* para)
{
	// weights 2D histogram from Data/MC of Lb->Jpsi pK 
	TFile *fw = new TFile(TString(para));
	//put mc name here and update to add weight of p&pT
	//TFile *f = new TFile(TString(output),"update");
    TFile *f = new TFile(TString(input));
	TH2D *w2d = (TH2D*)fw->Get("h3100");
	for(int i=0; i<5; ++i) {
		for(int j=0; j<5; ++j) {
			w[i][j] = w2d->GetBinContent(i+1,j+1);
		}
	}
	TTree* t_tmp = (TTree*)f->Get("DecayTree");
    TFile *fo = new TFile(TString(output),"recreate");
    TTree* t = t_tmp->CopyTree("1");
	double pt, px,py,pz, p, wppt;
	t->SetBranchAddress("B_TRUEPT", &pt);
	t->SetBranchAddress("B_TRUEP_X", &px);
	t->SetBranchAddress("B_TRUEP_Y", &py);
	t->SetBranchAddress("B_TRUEP_Z", &pz);
	TBranch *newBranch = t->Branch("wppt", &wppt, "wppt/D");
	Long64_t nentries = t->GetEntriesFast();
	//  nentries =10000;
	Long64_t nbytes = 0, nb = 0;

	for (Long64_t jentry=0; jentry<nentries;jentry++) {
		nb = t->GetEntry(jentry);   nbytes += nb;

		p = sqrt(px*px+py*py+pz*pz);
		wppt = weight(p, pt);
		newBranch->Fill();
	}

	//t->Write("",TObject::kOverwrite);
    t->Write();
	fo->Close(); 

}
