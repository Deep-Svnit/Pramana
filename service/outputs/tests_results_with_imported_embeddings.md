# PowerMind RAG service/tests.json Results With Imported Embeddings

## Q1

**Question:** The ANIL slide shows a manufacturing facility image — what type of product appears to be manufactured there based on the visual?

**Answer:** Not found in the document.

**Fallback:** True

**Citations:** [p1:x9], [p5:x121], [p5:c72], [p5:c75], [p28:c398], [p14:c221], [p24:c356], [p25:c892], [p30:x1196], [p27:x1096]

## Q2

**Question:** What is the Adani Family's equity stake in AEL as shown in the portfolio structure diagram?

**Answer:** The Adani Family's equity stake in AEL is 73.97% [p6:x185].

**Fallback:** False

**Citations:** [p6:c94], [p6:x135], [p6:x152], [p6:c90], [p6:x185], [p6:x191], [p6:x155], [p6:x136], [p4:c125], [p6:c89]

## Q3

**Question:** According to the ARTL Roads construction activity table, what was the lane-km construction volume in Q2-26 and what was the year-on-year change?

**Answer:** The lane-km construction volume in Q2-26 was 456.1 [p19:c284], and the year-on-year change was 153% [p19:c284].

**Fallback:** False

**Citations:** [p19:c284], [p20:c302], [p19:x833], [p20:c303], [p19:x809], [p20:x873], [p2:x48], [p19:x842], [p19:x810], [p19:x808]

## Q4

**Question:** Navi Mumbai Airport was just inaugurated — how soon can I expect it to contribute to the company's revenues?

**Answer:** Navi Mumbai Airport is set to commence operations from Q3 FY26 [p13:c206]. It is expected to contribute to the company's revenues, with large infra assets, including Navi Mumbai Airport, expected to contribute EBITDA from Q4 FY26 [p3:x105]. The airport was inaugurated on 08th October 2025 [p13:c206].

**Fallback:** False

**Citations:** [p13:c206], [p13:c201], [p3:c108], [p13:x490], [p13:x519], [p1:x17], [p3:x77], [p1:c34], [p1:c19], [p3:x105]

## Q5

**Question:** What is the total income of incubating business of H1-25?

**Answer:** The total income of incubating business of H1-25 is ₹17,287 [p2:x50].

**Fallback:** False

**Citations:** [p2:x50], [p2:x36], [p2:x41], [p2:x38], [p22:c328], [p2:x57], [p2:c17], [p2:x31], [p23:x969], [p2:c13]

## Q6

**Question:** In the Adani Portfolio overview infographic on page 6, under which exact parent category column is 'NDTV' placed, and what is the exact promoter holding percentage shown next to it?

**Answer:** 'NDTV' is placed under the parent category column 'Media' is not found, however 'NDTV' is placed under the parent category column 'Materials' is not found, but 'NDTV' is placed under a column and the exact promoter holding percentage shown next to it is 64.71% [p6:c103]. The parent category for 'NDTV' is actually 'Media' is not found, but 'NDTV' is under a category that includes 'Materials' and other categories such as 'Energy & Utility', 'Transport & Logistics' [p6:c103]. The exact parent category column is 'Media' is not found, but 'NDTV' is placed under 'Materials,' [p6:c103]. 

The exact parent category column is 'Materials,' [p6:c103] and the exact promoter holding percentage shown next to 'NDTV' is 64.71% [p6:c103].

**Fallback:** False

**Citations:** [p5:x130], [p5:c70], [p6:c103], [p6:c90], [p6:c89], [p5:x120], [p5:x132], [p6:c105], [p5:x125], [p5:x122]

## Q7

**Question:** What are the major business segments discussed in the document?

**Answer:** The major business segments discussed in the document are: business segments [p3:c51], consolidated income [p3:c51], EBITDA drivers [p3:c51], and airport performance [p3:c51].

**Fallback:** False

**Citations:** [p1:c5], [p3:c51], [p1:c7], [p31:c1030], [p7:c129], [p9:c160], [p16:c248], [p15:c234], [p18:c274], [p32:x643]

## Q8

**Question:** What is the consolidated total income in H1-26?

**Answer:** The consolidated total income in H1-26 is 49,263 [p22:c328] or 44,281 [p22:c328] (the context provides two different values for total income in H1-26, but it is unclear which one is the consolidated total income). Another source states the consolidated total income in H1-26 as 5,882 [p3:x91].

**Fallback:** False

**Citations:** [p4:c245], [p22:c328], [p21:c326], [p3:x91], [p16:c605], [p3:c37], [p3:x94], [p3:x88], [p23:x969], [p2:c13]

## Q9

**Question:** What drivers are mentioned for EBITDA changes in H1-26?

**Answer:** The drivers mentioned for EBITDA changes in H1-26 are Operational Efficiency [p15:c236] and Cost Management [p15:c236]. Additionally, Fuel cost reduction (+15%) [p21:c736] and occupancy rate (82% in H1) [p21:c736] are also mentioned as key drivers. Furthermore, the EBITDA drivers include Passenger Segment [p29:c417], Cargo Segment [p29:c417], and Other Segment [p29:c417], as well as Aviation [p18:c663], Logistics [p18:c663], and Other [p18:c663].

**Fallback:** False

**Citations:** [p15:c236], [p21:c736], [p14:c559], [p1:c5], [p21:c321], [p29:c417], [p18:c663], [p1:c7], [p2:c96], [p2:c24]

## Q10

**Question:** What is the CEO's email address?

**Answer:** Not found in the document.

**Fallback:** True

**Citations:** [p12:c477], [p24:c811], [p3:c114], [p24:c786], [p26:c905], [p26:c919], [p23:c770], [p1:x29], [p2:x52], [p3:x92]

## Q11

**Question:** Summarize airport performance in H1-26.

**Answer:** Airport performance in H1-26 includes a passenger traffic of 123.4M [p4:c66] and a revenue per passenger of $X.XX [p4:c66]. The occupancy rate was 85.2% [p4:c66]. Additionally, the airport EBITDA growth was +7% [p4:c244] and the EBITDA per passenger was $500 [p4:c244]. The average growth in airport performance was 13% [p13:c219].

**Fallback:** False

**Citations:** [p6:c112], [p1:c5], [p4:c244], [p4:c66], [p6:c106], [p7:c127], [p8:c148], [p13:c538], [p13:c219], [p15:c237]

## Q12

**Question:** Break that down into passenger and cargo changes.

**Answer:** To break down the changes into passenger and cargo, we can look at the provided data [p25:c901]. The passenger traffic change is an increase of ↑12% with a value of 120M [p25:c901]. The cargo volume change is an increase of ↑8% with a value of 5.2M tons [p25:c901]. 

In terms of revenue, the passenger segment revenue for Q2 FY26 is $900M [p31:c444] and the cargo segment revenue for Q2 FY26 is $850M [p31:c444]. The year-over-year growth for passenger revenue is 2% [p31:c444] and for cargo revenue is 3% [p31:c444]. 

EBITDA for the passenger segment in Q2 FY26 is $4.2B [p29:c417] and for the cargo segment in Q2 FY26 is $1.1B [p29:c417].

**Fallback:** False

**Citations:** [p29:c414], [p16:x663], [p27:c950], [p25:c897], [p7:c340], [p25:c901], [p29:c417], [p31:c444], [p31:c445], [p13:c218]

