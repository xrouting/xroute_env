module ispd18_sample ();

   // Internal wires
   wire net1230;
   wire net1238;
   wire net1235;
   wire net1239;
   wire net1231;
   wire net1232;
   wire net1234;
   wire net1236;
   wire net1233;
   wire net1240;
   wire net1237;

   // Assignments 

   // Module instantiations
   AO22XL inst7234 (
	   .Y (net1230) );
   AOI221X1 inst5195 (
	   .C0 (net1230) );
   NAND4X1 inst6050 (
	   .A (net1233) );
   AOI221X2 inst6458 (
	   .Y (net1234) );
   NAND3X1 inst5821 (
	   .B (net1231) );
   AOI22X1 inst5275 (
	   .Y (net1231) );
   AO22XL inst6286 (
	   .Y (net1239) );
   BUFX6 inst5638 (
	   .A (net1237) );
   AOI221X1 inst5333 (
	   .C0 (net1239) );
   NAND4X1 inst4382 (
	   .C (net1232) );
   NAND4X1 inst4597 (
	   .B (net1234) );
   AOI22X2 inst4189 (
	   .Y (net1233) );
   NOR2X1 inst4678 (
	   .Y (net1237) );
   AOI222X1 inst4062 (
	   .Y (net1232) );
   NOR4X4 inst4183 (
	   .A (net1235) );
   NAND4X1 inst4132 (
	   .Y (net1235) );
   BUFX3 inst3444 (
	   .Y (net1238) );
   NOR4X2 inst3502 (
	   .A (net1240) );
   BUFX3 inst3428 (
	   .A (net1238) );
   OR4X1 inst2908 (
	   .D (net1236) );
   NAND4X2 inst2591 (
	   .Y (net1236) );
   NAND3X2 inst2015 (
	   .Y (net1240) );
endmodule

