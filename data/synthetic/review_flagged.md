# The 15 synthetic phishing emails the LLM review doubted

All 15 are labelled phishing. Decide for each: keep, drop, or rewrite. Scores are 0-1, higher = more likely phishing (Sonnet flags at 0.25 or above).

## 1. `syn-chatgpt-015` — malicious attachment

- Doubted by: both reviewers (ambiguous, ambiguous)
- Detector scores: Sonnet 0.58, Haiku 0.78, TF-IDF 0.01, DeBERTa 0.03
- Reviewer A1 (ambiguous): Labelled phishing but nothing in the text gives it away: consistent sender domain, ordinary .docx, no macros, no credentials, mild deadline. Reads the same as the legitimate work emails with attachments in this set (e.g. 053, 093, 098).
- Reviewer B2 (ambiguous): Labelled malicious attachment, but sender domain matches the organisation, the tone is relaxed, the deadline is mild and nothing fraudulent is requested; indistinguishable from a routine internal document review.

**From:** Medical Records Administration <records@clearwater-medical-group.example>  
**Subject:** Quick review needed: Updated patient transfer summary

```
Hi Daniel,

Hope everything is going well. We have completed the latest review of the patient transfer records from last week, and there is one document that still needs your attention before the records can be closed.

I've attached the file Transfer_Summary_Nov22.docx with the updated information from the records team. Could you please open it and check the highlighted sections when you have a moment? We need to make sure the receiving department information and the transfer dates are correct.

The review should only take a few minutes. If you notice any differences from your original notes, please add your comments directly in the document and send it back to me by tomorrow morning.

Thanks for helping us get this completed. Let me know if you have any questions.

Best,
Laura Chen
Records Administration
Clearwater Medical Group
```

## 2. `syn-chatgpt-037` — malicious attachment

- Doubted by: both reviewers (ambiguous, ambiguous)
- Detector scores: Sonnet 0.35, Haiku 0.22, TF-IDF 0.41, DeBERTa 0.05
- Reviewer A1 (ambiguous): Labelled phishing but indistinguishable from a genuine bank statement email: sender domain matches Harborstone Bank, plain .pdf, no pressure, no credentials or payment requested.
- Reviewer B1 (ambiguous): Sender domain matches Harborstone Bank, there is no link, no threat, no deadline, no credential or payment request; it reads as a routine statement email and nothing in the text marks it as malicious.

**From:** Document Services <documents@harborstone-bank.example>  
**Subject:** Your requested monthly account statement

```
Dear Ms. Carter,

The monthly account statement you requested has been prepared and is attached to this message as Account_Statement_November_2026.pdf.

The document contains the transaction summary and account activity for the period ending November 20, 2026. Please review the statement at your convenience and retain a copy for your records.

For confidentiality purposes, the statement is provided as a separate document rather than included in the email body. Please open the attached file to view the complete information.

If you did not request this statement, or if the document appears inconsistent with your records, please contact our document services department by replying to this email.

Thank you for banking with Harborstone Bank.

Regards,
Document Services
Harborstone Bank
```

## 3. `syn-chatgpt-062` — malicious attachment

- Doubted by: both reviewers (ambiguous, ambiguous)
- Detector scores: Sonnet 0.62, Haiku 0.22, TF-IDF 0.42, DeBERTa 0.07
- Reviewer A1 (ambiguous): Labelled phishing but nothing distinguishes it from a genuine statement notice: domain matches Granite Harbor Bank, plain .pdf, no urgency, no credentials or payment requested, points to 'your established banking channel'.
- Reviewer B1 (ambiguous): Sender domain matches Granite Harbor Bank, there is no link, no urgency, no threat, no credential or payment request, and it refers questions to the customer's established banking channel. Nothing in the text distinguishes it from a genuine statement email.

**From:** Account Services <documents@granite-harbor-bank.example>  
**Subject:** Your monthly account statement is available

```
Dear Customer,

Your November account statement has been generated and is ready for review.

The statement includes transaction activity through November 21, 2026, along with the applicable service charges and interest calculations. For your convenience, we have also included a summary of recent account maintenance activity.

Please open the attached document, November_Statement_2026.pdf, to review the information and retain a copy for your records.

If you have questions regarding any transaction or believe an entry requires correction, please contact our Account Services team through your established banking channel.

Thank you for choosing Granite Harbor Bank.

Sincerely,
Account Services Department
```

## 4. `syn-grok-033` — CEO fraud / business email compromise

- Doubted by: both reviewers (ambiguous, ambiguous)
- Detector scores: Sonnet 0.88, Haiku 0.88, TF-IDF 0.29, DeBERTa 0.04
- Reviewer A2 (ambiguous): Labelled phishing, but the sender domain is a plain company domain and the mail only asks a colleague to forward existing wire info to an assistant: no payment, bank-detail change, gift cards, link or credentials. A careful reader could take it as an ordinary urgent internal request.
- Reviewer B2 (ambiguous): Labelled BEC but asks only that existing wire info be forwarded; no payment, gift card or changed-bank-detail request, no organisation named, no identifiable impersonation. Reads like a terse (and slightly garbled) urgent colleague request; could go either way.

**From:** Maria Chen <maria.chen@riverstone-finance.example>  
**Subject:** Can you help with this?

```
Hi,

I am in the middle of a deal and need the updated bank details for the vendor payment sent to my assistant. Can you forward the current wire info for account ending 7742? It is urgent for closing today.

Thanks a lot,
Maria
```

## 5. `syn-grok-040` — CEO fraud / business email compromise

- Doubted by: both reviewers (ambiguous, ambiguous)
- Detector scores: Sonnet 0.82, Haiku 0.92, TF-IDF 0.31, DeBERTa 0.08
- Reviewer A2 (ambiguous): Labelled phishing, but the request is only to send vendor payment details; it does not ask for a payment, gift cards or a bank-detail change, and is internally incoherent. The '-exec' domain is the only tell, so the text could pass as a normal urgent colleague request.
- Reviewer B1 (ambiguous): The sender asks the recipient to SEND vendor payment details rather than to make a payment, buy gift cards or use changed bank details, and the text is internally confused about who is switching banks. Only the '-exec' sender domain hints at fraud; there is no organisation named to compare it against.

**From:** Sam Patel <sam.patel@clearbooks-exec.example>  
**Subject:** Need those numbers today

```
Hi,

Can you send over the updated vendor payment details for the Q3 close? We're switching banks and I need the new wire info ASAP so I can process the transfer this afternoon.

Thanks,
Sam
```

## 6. `syn-grok-073` — malicious attachment

- Doubted by: both reviewers (label_wrong, label_wrong)
- Detector scores: Sonnet 0.22, Haiku 0.22, TF-IDF 0.03, DeBERTa 0.03
- Reviewer A3 (label_wrong): No fraudulent element: sender and link are both on the organisation's own domain, there is no pressure, no credential or payment request; it reads as a normal support reply sharing a report.
- Reviewer B3 (label_wrong): No fraudulent element: sender and link are both on nimbusflow-platforms.example, matching the signature; no urgency, credentials or payment. Reads as a normal support reply sharing a report.

**From:** Support Team <support@nimbusflow-platforms.example>  
**Subject:** Shared file: your requested usage report

```
Hi,

Thanks for getting in touch earlier. I have pulled the usage report you asked for and placed it in the shared folder. You can open the document Usage_Report_November.pdf directly from this link:

https://files.nimbusflow-platforms.example/share/usage-nov

It covers the last 30 days and includes the breakdown by team. Let me know if you need any columns added or a different date range and I will regenerate it.

Happy to help with anything else.

Best,
Support Team
NimbusFlow Platforms
```

## 7. `syn-grok-075` — CEO fraud / business email compromise

- Doubted by: both reviewers (label_wrong, label_wrong)
- Detector scores: Sonnet 0.72, Haiku 0.88, TF-IDF 0.71, DeBERTa 0.04
- Reviewer A3 (label_wrong): Labelled BEC but requests no payment and gives no bank details; it actually tells staff to hold transfers and wait for written confirmation, from the organisation's own domain. Reads as a legitimate finance control notice.
- Reviewer B3 (label_wrong): No fraudulent request: sender domain matches NimbusFlow Platforms, no bank details are given, and the email tells staff to HOLD transfers and await written confirmation, which is a sound control. Reads as a legitimate finance notice.

**From:** CFO Office <cfo-office@nimbusflow-platforms.example>  
**Subject:** Vendor bank detail change – confirmation required

```
Team,

Please be advised that our primary cloud infrastructure vendor has updated their payment instructions. All future invoices should be settled using the new account details that will be circulated separately by the finance systems team.

Until the formal notice is issued, hold any pending transfers and await written confirmation from this office. Reply to acknowledge receipt so we can track compliance.

Thank you for your attention to this control.

CFO Office
NimbusFlow Platforms
```

## 8. `syn-grok-085` — malicious attachment

- Doubted by: both reviewers (label_wrong, ambiguous)
- Detector scores: Sonnet 0.12, Haiku 0.08, TF-IDF 0.00, DeBERTa 0.04
- Reviewer A3 (label_wrong): No fraudulent element: sender domain matches Ridgeway Logistics, tone is relaxed ('when you get a chance'), no pressure, no request beyond reading a schedule. Indistinguishable from an ordinary work email with an attachment.
- Reviewer B2 (ambiguous): Labelled malicious attachment, but sender domain matches the organisation and the message is a casual, no-pressure schedule update; nothing a careful reader could point to as fraudulent.

**From:** Sam Okonkwo <sam.okonkwo@ridgewaylogistics.example>  
**Subject:** Updated delivery schedule attached

```
Hi,

Hope you’re well. I’ve attached the revised delivery schedule for next week’s northern route. A couple of the drop-off windows shifted because of the roadworks, so please have a look when you get a chance.

The file is called Ridgeway_Northern_Route_Rev3.pdf. Let me know if any of the new times clash with your side.

Thanks,
Sam
Ridgeway Logistics
```

## 9. `syn-grok-144` — malicious attachment

- Doubted by: both reviewers (ambiguous, ambiguous)
- Detector scores: Sonnet 0.72, Haiku 0.25, TF-IDF 0.21, DeBERTa 0.08
- Reviewer A3 (ambiguous): Nothing in the text gives it away: the sender domain is consistent with Northern Star Airlines, there is no urgency, threat, link or request. An airline emailing a boarding pass PDF is normal, so a careful reader could easily take it as legitimate.
- Reviewer B2 (ambiguous): Labelled malicious attachment, but it is a three-sentence boarding-pass email from a domain consistent with the airline, with no pressure, no pretext and no request; text alone gives nothing away.

**From:** Northern Star Airlines <documents@northernstar-air.example>  
**Subject:** Boarding documents attached

```
Dear passenger,

Please find attached your updated boarding pass and itinerary as BoardingPass_NS441.pdf. Open the file to view gate and seat details. Contact us if the attachment does not open.
```

## 10. `syn-chatgpt-101` — malicious attachment

- Doubted by: one of two reviewers (ok, ambiguous)
- Detector scores: Sonnet 0.62, Haiku 0.72, TF-IDF 0.00, DeBERTa 0.03
- Reviewer A2 (ok): only tell is the lookalike sender domain plus a push to open a named attachment; body alone reads as a normal colleague mail (weak but fits spec).
- Reviewer B2 (ambiguous): Labelled malicious attachment; only possible tell is the '-billing' suffix on the sender domain for an administrative coordinator, which is weak given legit emails in this set use similar suffixed domains. Content is a calm, plausible internal document check with a Thursday deadline.

**From:** Mara Ellison <m.ellison@westbridge-university-billing.example>  
**Subject:** Revised procurement documentation

```
Hello Daniel,

The procurement office has returned the documentation for the equipment order submitted last week. There is one revised page that needs to be reviewed before the request can move forward.

I've attached the updated file as Procurement_Revision_17.pdf. Could you please open it and confirm that the figures on page 2 match the original submission? The office has asked us to resolve this before their next processing run on Thursday.

I was told the previous version had a formatting issue, so please use this copy rather than the one circulated earlier. Let me know if you notice anything else that needs correcting.

Thanks,
Mara Ellison
Administrative Coordinator
Westbridge University
```

## 11. `syn-chatgpt-122` — malicious attachment

- Doubted by: one of two reviewers (ok, ambiguous)
- Detector scores: Sonnet 0.62, Haiku 0.62, TF-IDF 0.04, DeBERTa 0.03
- Reviewer A2 (ok): lookalike '-docs' sender domain pushing a named attachment; body alone reads as benign (weak but fits spec).
- Reviewer B3 (ambiguous): Reads as a routine internal request to check a PDF report; no organisation is named in the body, no threat, no credentials or payment. The only tell is the '-docs' suffix on the sender domain.

**From:** Production Records <records@ironvale-manufacturing-docs.example>  
**Subject:** Updated inspection report for review

```
Hello Martin,

Please find attached the revised inspection report from the November production review. The document includes the corrections discussed with the quality team and the updated measurements for the final batch.

Attachment: Batch_Inspection_2026_11_Final.pdf

Could you please open the document and confirm the figures in the second section? We need to ensure the report is consistent with the records already submitted by your department before it is included in the monthly production archive.

The previous version contained some incorrect measurements, so please refer only to this updated copy. It would be helpful to receive your confirmation by tomorrow morning, as the archive is scheduled for completion shortly afterwards.

Thank you for your assistance.

Regards,
Petra Weiss
Production Records Coordinator
```

## 12. `syn-grok-029` — CEO fraud / business email compromise

- Doubted by: one of two reviewers (ok, ambiguous)
- Detector scores: Sonnet 0.95, Haiku 0.92, TF-IDF 0.13, DeBERTa 0.06
- Reviewer A2 (ok): gift-card request from a boss 'in meetings all day' on an '-exec' domain. Weak: it never asks for the codes and says to 'expense them', so the body alone could pass as a real errand.
- Reviewer B3 (ambiguous): Gift-card request from an '-exec' domain, but it only asks to buy cards 'from the usual place' for a client event and expense them; no codes to send, no secrecy, no remote handover. Could be a real boss's errand.

**From:** Jordan Hale <jordan.hale@vividmart-exec.example>  
**Subject:** Quick favor

```
Hey,

I'm in meetings all day and need you to pick up some gift cards for the client event tomorrow. Grab 10 of the $50 ones from the usual place and expense them. Can you handle that today?

Thanks,
Jordan
```

## 13. `syn-grok-064` — CEO fraud / business email compromise

- Doubted by: one of two reviewers (ambiguous, ok)
- Detector scores: Sonnet 0.93, Haiku 0.92, TF-IDF 0.50, DeBERTa 0.04
- Reviewer A3 (ambiguous): Bank-detail change is the only fraud marker; the sender domain matches the organisation, there is no urgency or secrecy, and it points to the intranet policy document and 'usual channels' for verification, so it reads like a genuine internal treasury notice.
- Reviewer B1 (ok): Unsigned 'Finance Controller' instructs staff to redirect salary and vendor payments to a new bank account and confirm by reply.

**From:** Finance Controller <controller@talentpay-services.example>  
**Subject:** Updated banking instructions for next payroll cycle

```
Colleagues,

Please note that effective with the next payroll run we are moving the primary disbursement account to a new banking relationship in order to improve settlement times and reduce fees.

All future salary and vendor payments that previously used the old account ending 4471 should now be directed to:

Bank name: Northbridge Commercial Bank
Account name: TalentPay Services Ltd
Account number: 883920174
Sort code: 20-45-67

Kindly update your records and confirm receipt of this instruction by replying to this message. The change has already been approved by the board and is reflected in the latest treasury policy document available on the intranet.

No other process changes are required. If you have any questions regarding the transition, contact the finance team through the usual channels.

Regards,
Finance Controller
TalentPay Services
```

## 14. `syn-grok-068` — CEO fraud / business email compromise

- Doubted by: one of two reviewers (ok, ambiguous)
- Detector scores: Sonnet 0.88, Haiku 0.92, TF-IDF 0.04, DeBERTa 0.03
- Reviewer A3 (ok): Gift-card purchase request with 'tied up in meetings' pretext from a '-finance' lookalike domain.
- Reviewer B3 (ambiguous): Asks a colleague to buy $500 of gift cards and bring them physically to the office against a receipt and budget code; no codes sent, no secrecy. Only tell is the domain harborview-civic-finance.example for Harborview Municipal Services.

**From:** Alex Rivera <a.rivera@harborview-civic-finance.example>  
**Subject:** Gift cards for the community event – can you help?

```
Hi,

Hope you are having a good week. I am organising the small thank-you gifts for the volunteers at next month’s community clean-up and the usual supplier is out of stock.

Could you pick up $500 in gift cards from the local supermarket today and bring them to the office? I will reimburse you straight away from the events budget once I have the receipt. Just grab whatever is available – grocery or general merchandise is fine.

I am tied up in back-to-back meetings until late, otherwise I would do it myself. Let me know if you can manage it and I will send the exact budget code.

Thanks a lot,
Alex Rivera
Community Engagement
Harborview Municipal Services
```

## 15. `syn-grok-120` — malicious attachment

- Doubted by: one of two reviewers (ok, ambiguous)
- Detector scores: Sonnet 0.72, Haiku 0.62, TF-IDF 0.03, DeBERTa 0.07
- Reviewer A3 (ok): Pushes a named attachment from riverside-cityworks-docs.example, a lookalike of the riverside-cityworks.example domain. Subtle: the domain suffix is the only giveaway.
- Reviewer B3 (ambiguous): Three-line casual note attaching a PDF 'you asked for'; no threat, no credentials, no payment. The only tell is the '-docs' suffix in the sender domain.

**From:** Riverside City Works <permits@riverside-cityworks-docs.example>  
**Subject:** Scan of your permit

```
Hey,

Attached is the scan of the updated permit you asked for. File name is Riverside_Permit_Update.pdf. Open it and let me know if the dates look right.

Thanks
```
