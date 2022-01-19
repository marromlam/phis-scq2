
void get_oscomb_tagger(
	const char* tree1,
	const char* tree2,
	const char* tree3,
	const char* offile
	)
{
   TChain *chain1 = new TChain();
   TChain *chain2 = new TChain();
   TChain *chain3 = new TChain();
   chain1->Add( tree1 );
   chain2->Add( tree2 );
   chain3->Add( tree3 );

   int    tagos_dec_new;
   double tagos_eta_new;
   int    tagos_dec_old;
   double tagos_eta_old;

   chain2->SetBranchAddress("OS_Combination_DEC",&tagos_dec_new);
   chain2->SetBranchAddress("OS_Combination_ETA",&tagos_eta_new);
   chain3->SetBranchAddress("OS_Combination_DEC",&tagos_dec_old);
   chain3->SetBranchAddress("OS_Combination_ETA",&tagos_eta_old);

   TFile *rootfile = new TFile( offile, "recreate" );
   TTree *DecayTree = chain1->CloneTree(0);
   DecayTree->Branch("tagos_dec_new",&tagos_dec_new,"tagos_dec_new/I");
   DecayTree->Branch("tagos_eta_new",&tagos_eta_new,"tagos_eta_new/D");
   DecayTree->Branch("tagos_dec_old",&tagos_dec_old,"tagos_dec_old/I");
   DecayTree->Branch("tagos_eta_old",&tagos_eta_old,"tagos_eta_old/D");

   cout<<"======================================"<<endl;
   cout<<" Input tree1 : "<<tree1<<", "<<chain1->GetEntries()<<endl;
   cout<<" Input tree2 : "<<tree2<<", "<<chain2->GetEntries()<<endl;
   cout<<" Input tree3 : "<<tree3<<", "<<chain3->GetEntries()<<endl;
   cout<<" Output root : "<<offile<<endl;
   cout<<"======================================"<<endl;

   for( int i=0; i<chain1->GetEntries(); i++ ){
	if(i%10000==0) cout<<"  Processing "<<i/1000<<" k. "<<endl;
	chain1->GetEntry(i);
	chain2->GetEntry(i);
	chain3->GetEntry(i);
	DecayTree->Fill();
   }

   rootfile->cd();
   DecayTree->Fill();
   rootfile->Close();

}
