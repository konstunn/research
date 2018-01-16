add_cus_dep('pytxcode', 'tex', 0, 'pythontex');
sub pythontex { return system("pythontex3 \"$_[0]\""); }
