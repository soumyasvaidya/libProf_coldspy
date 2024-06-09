import markdown
import base64

base64_text = "IVtPcGVuUGl0b24gTG9nb10oL2RvY3Mvb3BlbnBpdG9uX2xvZ29fYmxhY2sucG5nP3Jhdz10cnVlKQoKIyBPcGVuUGl0b24gUmVzZWFyY2ggUGxhdGZvcm0gICBbIVtCdWlsZCBTdGF0dXNdKGh0dHBzOi8vamVua2lucy5wcmluY2V0b24uZWR1L2J1aWxkU3RhdHVzL2ljb24/am9iPWNsb3VkL3BpdG9uX2dpdF9wdXNoX21hc3RlcildKGh0dHBzOi8vamVua2lucy5wcmluY2V0b24uZWR1L2pvYi9jbG91ZC9qb2IvcGl0b25fZ2l0X3B1c2hfbWFzdGVyLykKCk9wZW5QaXRvbiBpcyB0aGUgd29ybGQncyBmaXJzdCBvcGVuIHNvdXJjZSwgZ2VuZXJhbCBwdXJwb3NlLCBtdWx0aXRocmVhZGVkIG1hbnljb3JlIHByb2Nlc3Nvci4gSXQgaXMgYSB0aWxlZCBtYW55Y29yZSBmcmFtZXdvcmsgc2NhbGFibGUgZnJvbSBvbmUgdG8gMS8yIGJpbGxpb24gY29yZXMuIEl0IGlzIGEgNjQtYml0IGFyY2hpdGVjdHVyZSB1c2luZyBTUEFSQyB2OSBJU0Egd2l0aCBhIGRpc3RyaWJ1dGVkIGRpcmVjdG9yeS1iYXNlZCBjYWNoZSBjb2hlcmVuY2UgcHJvdG9jb2wgYWNyb3NzIG9uLWNoaXAgbmV0d29ya3MuIEl0IGlzIGhpZ2hseSBjb25maWd1cmFibGUgaW4gYm90aCBjb3JlIGFuZCB1bmNvcmUgY29tcG9uZW50cy4gT3BlblBpdG9uIGhhcyBiZWVuIHZlcmlmaWVkIGluIGJvdGggQVNJQyBhbmQgbXVsdGlwbGUgWGlsaW54IEZQR0EgcHJvdG90eXBlcyBydW5uaW5nIGZ1bGwtc3RhY2sgRGViaWFuIGxpbnV4LiBXZSBoYXZlIHJlbGVhc2VkIGJvdGggdGhlIFZlcmlsb2cgUlRMIGNvZGUgYXMgd2VsbCBhcyBzeW50aGVzaXMgYW5kIGJhY2stZW5kIGZsb3cuIFdlIGJlbGlldmUgT3BlblBpdG9uIGlzIGEgZ3JlYXQgZnJhbWV3b3JrIGZvciByZXNlYXJjaGVycyBpbiBjb21wdXRlciBhcmNoaXRlY3R1cmUsIE9TLCBjb21waWxlcnMsIEVEQSwgc2VjdXJpdHkgYW5kIG1vcmUuCgpPcGVuUGl0b24gaGFzIGJlZW4gcHVibGlzaGVkIGluIEFTUExPUyAyMDE2OiBKb25hdGhhbiBCYWxraW5kLCBNaWNoYWVsIE1jS2Vvd24sIFlhb3NoZW5nIEZ1LCBUcmkgTmd1eWVuLCBZYW5xaSBaaG91LCBBbGV4ZXkgTGF2cm92LCBNb2hhbW1hZCBTaGFocmFkLCBBZGkgRnVjaHMsIFNhbXVlbCBQYXluZSwgWGlhb2h1YSBMaWFuZywgTWF0dGhldyBNYXRsLCBhbmQgRGF2aWQgV2VudHpsYWZmLiAiT3BlblBpdG9uOiBBbiBPcGVuIFNvdXJjZSBNYW55Y29yZSBSZXNlYXJjaCBGcmFtZXdvcmsuIiBJbiBQcm9jZWVkaW5ncyBvZiB0aGUgMjFzdCBJbnRlcm5hdGlvbmFsIENvbmZlcmVuY2Ugb24gQXJjaGl0ZWN0dXJhbCBTdXBwb3J0IGZvciBQcm9ncmFtbWluZyBMYW5ndWFnZXMgYW5kIE9wZXJhdGluZyBTeXN0ZW1zIChBU1BMT1MgJzE2KSwgQXByaWwgMjAxNi4KCiMjIyMgRmluZCBvdXQgbW9yZQoKLSBNb3JlIGluZm9ybWF0aW9uIGFib3V0IE9wZW5QaXRvbiBpcyBhdmFpbGFibGUgYXQgd3d3Lm9wZW5waXRvbi5vcmcKLSBbRm9sbG93IHVzIG9uIFR3aXR0ZXIhXShodHRwczovL3d3dy50d2l0dGVyLmNvbS9vcGVucGl0b24pCi0gR2V0IGhlbHAgZnJvbSBvdGhlcnMgYnkgam9pbmluZyBvdXIgW0dvb2dsZSBHcm91cF0oaHR0cHM6Ly9ncm91cHMuZ29vZ2xlLmNvbS9ncm91cC9vcGVucGl0b24pCi0gS2VlcCB1cC10by1kYXRlIHdpdGggdGhlIGxhdGVzdCByZWxlYXNlcyBhdCB0aGUgW09wZW5QaXRvbiBCbG9nXShodHRwczovL29wZW5waXRvbi1ibG9nLnByaW5jZXRvbi5lZHUpCgpJZiB5b3UgdXNlIE9wZW5QaXRvbiBpbiB5b3VyIHJlc2VhcmNoIHBsZWFzZSByZWZlcmVuY2Ugb3VyIEFTUExPUyAyMDE2IHBhcGVyIG1lbnRpb25lZCBhYm92ZSBhbmQgc2VuZCB1cyBhIGNpdGF0aW9uIG9mIHlvdXIgd29yay4KCiMjIyMgRG9jdW1lbnRhdGlvbgoKVGhlcmUgYXJlIHNldmVyYWwgZGV0YWlsZWQgcGllY2VzIG9mIGRvY3VtZW50YXRpb24gYWJvdXQgT3BlblBpdG9uIGluIHRoZSBkb2NzIGZvbGRlciBsaXN0ZWQgYmVsb3c6CgotIFtPcGVuUGl0b24gU2ltdWxhdGlvbiBNYW51YWxdKGh0dHA6Ly9wYXJhbGxlbC5wcmluY2V0b24uZWR1L29wZW5waXRvbi9kb2NzL3NpbV9tYW4ucGRmKQotIFtPcGVuUGl0b24gTWljcm9hcmNoaXRlY3R1cmUgU3BlY2lmaWNhdGlvbl0oaHR0cDovL3BhcmFsbGVsLnByaW5jZXRvbi5lZHUvb3BlbnBpdG9uL2RvY3MvbWljcm9fYXJjaC5wZGYpCi0gW09wZW5QaXRvbiBGUEdBIFByb3RvdHlwZSBNYW51YWxdKGh0dHA6Ly9wYXJhbGxlbC5wcmluY2V0b24uZWR1L29wZW5waXRvbi9kb2NzL2ZwZ2FfbWFuLnBkZikKLSBbT3BlblBpdG9uIFN5bnRoZXNpcyBhbmQgQmFjay1lbmQgTWFudWFsXShodHRwOi8vcGFyYWxsZWwucHJpbmNldG9uLmVkdS9vcGVucGl0b24vZG9jcy9zeW5iY2tfbWFuLnBkZikKCldlIGFsc28gaG9zdCBHaXRIdWIgcmVwb3NpdG9yaWVzIGZvciBvdGhlciBwYXJ0cyBvZiB0aGUgcHJvamVjdCwgaW5jbHVkaW5nOgoKLSBbUGl0b24gTGludXggS2VybmVsXShodHRwczovL2dpdGh1Yi5jb20vUHJpbmNldG9uVW5pdmVyc2l0eS9waXRvbi1saW51eCkKLSBbUGl0b24gSHlwZXJ2aXNvcl0oaHR0cHM6Ly9naXRodWIuY29tL1ByaW5jZXRvblVuaXZlcnNpdHkvcGl0b24tc3cpCgojIyMjIEVudmlyb25tZW50IFNldHVwCi0gVGhlIGBgYFBJVE9OX1JPT1RgYGAgZW52aXJvbm1lbnQgdmFyaWFibGUgc2hvdWxkIHBvaW50IHRvIHRoZSByb290IG9mIHRoZSBPcGVuUGl0b24gcmVwb3NpdG9yeQotIFRoZSBTeW5vcHN5cyBlbnZpcm9ubWVudCBmb3Igc2ltdWxhdGlvbiBzaG91bGQgYmUgc2V0dXAgc2VwYXJhdGVseSBieSB0aGUgdXNlci4gIEJlc2lkZXMgYWRkaW5nIGNvcnJlY3QgcGF0aHMgdG8geW91ciBgYGBQQVRIYGBgIGFuZCBgYGBMRF9MSUJSQVJZX1BBVEhgYGAgKHVzdWFsbHkgYWNjb21wbGlzaGVkIGJ5IGEgc2NyaXB0IHByb3ZpZGVkIGJ5IFN5bm9wc3lzKSwgdGhlIE9wZW5QaXRvbiB0b29scyBzcGVjaWZpY2FsbHkgcmVmZXJlbmNlIHRoZSBgYGBWQ1NfSE9NRWBgYCBlbnZpcm9ubWVudCB2YXJpYWJsZSB3aGljaCBzaG91bGQgICBwb2ludCB0byB0aGUgcm9vdCBvZiB0aGUgU3lub3BzeXMgVkNTIGluc3RhbGxhdGlvbi4KICAgIC0gKipOb3RlKio6IERlcGVuZGluZyBvbiB5b3VyIHN5c3RlbSBzZXR1cCwgU3lub3BzeXMgdG9vbHMgbWF5IHJlcXVpcmUgdGhlIGBgYC1mdWxsNjRgYGAgZmxhZy4gIFRoaXMgY2FuIGVhc2lseSBiZSBhY2NvbXBsaXNoZWQgYnkgYWRkaW5nIGEgYmFzaCBmdW5jdGlvbiBhcyBzaG93biBpbiB0aGUgZm9sbG93aW5nIGV4YW1wbGUgZm9yIFZDUyAoYWxzbyByZXF1aXJlZCBmb3IgVVJHKToKCiAgICAgICAgYGBgYmFzaAogICAgICAgIGZ1bmN0aW9uIHZjcygpIHsgY29tbWFuZCB2Y3MgLWZ1bGw2NCAiJEAiOyB9OyBleHBvcnQgLWYgdmNzCiAgICAgICAgYGBgCgotIFJ1biBgYGBzb3VyY2UgJFBJVE9OX1JPT1QvcGl0b24vcGl0b25fc2V0dGluZ3MuYmFzaGBgYCB0byBzZXR1cCB0aGUgZW52aXJvbm1lbnQKICAgIC0gQSBDU2hlbGwgdmVyc2lvbiBvZiB0aGlzIHNjcmlwdCBpcyBwcm92aWRlZCwgYnV0IE9wZW5QaXRvbiBoYXMgbm90IGJlZW4gdGVzdGVkIGZvciBhbmQgY3VycmVudGx5IGRvZXMgbm90IHN1cHBvcnQgQ1NoZWxsCgotIE5vdGU6IE9uIG1hbnkgc3lzdGVtcywgeW91IG11c3QgcnVuIHRoZSBgYGBta3Rvb2xzYGBgIGNvbW1hbmQgb25jZSB0byByZWJ1aWxkIGEgbnVtYmVyIG9mIHRoZSB0b29scyBiZWZvcmUgY29udGludWluZy4gSWYgeW91IHNlZSBpc3N1ZXMgbGF0ZXIgd2l0aCBidWlsZGluZyBvciBydW5uaW5nIHNpbXVsYXRpb25zLCB0cnkgcnVubmluZyBgYGBta3Rvb2xzYGBgIGlmIHlvdSBoYXZlIG5vdCBhbHJlYWR5LgoKLSBUb3AgbGV2ZWwgZGlyZWN0b3J5IHN0cnVjdHVyZToKICAgIC0gcGl0b24vCiAgICAgICAgLSBBbGwgT3BlblBpdG9uIGRlc2lnbiBhbmQgdmVyaWZpY2F0aW9uIGZpbGVzCiAgICAtIGRvY3MvCiAgICAgICAgLSBPcGVuUGl0b24gZG9jdW1lbnRhdGlvbgogICAgLSBidWlsZC8KICAgICAgICAtIFdvcmtpbmcgZGlyZWN0b3J5IGZvciBzaW11bGF0aW9uIGFuZCBzaW11bGF0aW9uIG1vZGVscwoKPT09PT09PT09PT09PT09PT09PT09PT09PT0KCiMjIyMgQnVpbGRpbmcgYSBzaW11bGF0aW9uIG1vZGVsCjEuIGBgYGNkICRQSVRPTl9ST09UL2J1aWxkYGBgCjIuIGBgYHNpbXMgLXN5cz1tYW55Y29yZSAteF90aWxlcz0xIC15X3RpbGVzPTEgLXZjc19idWlsZGBgYCBidWlsZHMgYSBzaW5nbGUgdGlsZSBPcGVuUGl0b24gc2ltdWxhdGlvbiBtb2RlbC4KMy4gQSBkaXJlY3RvcnkgZm9yIHRoZSBzaW11bGF0aW9uIG1vZGVsIHdpbGwgYmUgY3JlYXRlZCBpbiBgYGAkUElUT05fUk9PVC9idWlsZGBgYCBhbmQgdGhlIHNpbXVsYXRpb24gbW9kZWwgY2FuIG5vdyBiZSB1c2VkIHRvIHJ1biB0ZXN0cy4gIEZvciBtb3JlIGRldGFpbHMgb24gYnVpbGRpbmcgc2ltdWxhdGlvbiBtb2RlbHMsIHBsZWFzZSByZWZlciB0byB0aGUgT3BlblBpdG9uIGRvY3VtZW50YXRpb24uCgo+IE5vdGU6IGlmIHlvdSB3b3VsZCBsaWtlIHRvIGRlY3JlYXNlIHRoZSB0ZXN0YmVuY2ggbW9uaXRvciBvdXRwdXQgdG8gYSBtaW5pbXVtLCBhcHBlbmQgYC1jb25maWdfcnRsPU1JTklNQUxfTU9OSVRPUklOR2AgdG8geW91ciBidWlsZCBjb21tYW5kIGluIHN0ZXAgMi4gYWJvdmUuCgo9PT09PT09PT09PT09PT09PT09PT09PT09PQoKIyMjIyBSdW5uaW5nIGEgc2ltdWxhdGlvbgoxLiBgYGBjZCAkUElUT05fUk9PVC9idWlsZGBgYAoyLiBgYGBzaW1zIC1zeXM9bWFueWNvcmUgLXhfdGlsZXM9MSAteV90aWxlcz0xIC12Y3NfcnVuIHByaW5jZXRvbi10ZXN0LXRlc3Quc2BgYCBydW5zIGEgc2ltcGxlIGFycmF5IHN1bW1hdGlvbiB0ZXN0IGdpdmVuIHRoZSBzaW11bGF0aW9uIG1vZGVsIGlzIGFscmVhZHkgYnVpbHQuCjMuIFRoZSBzaW11bGF0aW9uIHdpbGwgcnVuIGFuZCBnZW5lcmF0ZSBtYW55IGxvZyBmaWxlcyBhbmQgc2ltdWxhdGlvbiBvdXRwdXQgdG8gc3Rkb3V0LiAgRm9yIG1vcmUgZGV0YWlscyBvbiBydW5uaW5nIGEgc2ltdWxhdGlvbiwgcHJvdmlkZWQgdGVzdHMvc2ltdWxhdGlvbnMgaW4gdGhlIHRlc3Qgc3VpdGUsIGFuZCB1bmRlcnN0YW5kaW5nIHRoZSBzaW11bGF0aW9uIGxvZyBmaWxlcyBhbmQgb3V0cHV0LCBwbGVhc2UgcmVmZXIgdG8gdGhlIE9wZW5QaXRvbiBkb2N1bWVudGF0aW9uLgoKPT09PT09PT09PT09PT09PT09PT09PT09PT0KCiMjIyMgUnVubmluZyBhIHJlZ3Jlc3Npb24KQSByZWdyZXNzaW9uIGlzIGEgc2V0IG9mIHNpbXVsYXRpb25zL3Rlc3RzIHdoaWNoIHJ1biBvbiB0aGUgc2FtZSBzaW11bGF0aW9uIG1vZGVsLgoKMS4gYGBgY2QgJFBJVE9OX1JPT1QvYnVpbGRgYGAKMi4gYGBgc2ltcyAtc2ltX3R5cGU9dmNzIC1ncm91cD10aWxlMV9taW5pYGBgIHJ1bnMgdGhlIHNpbXVsYXRpb25zIGluIHRoZSB0aWxlMV9taW5pIHJlZ3Jlc3Npb24gZ3JvdXAuCjMuIFRoZSBzaW11YXRpb24gbW9kZWwgd2lsbCBiZSBidWlsdCBhbmQgYWxsIHNpbXVsYXRpb25zIHdpbGwgYmUgcnVuIHNlcXVlbnRpYWxseS4gIEluIGFkZGl0aW9uIHRvIHRoZSBzaW11bGF0aW9uIG1vZGVsIGRpcmVjdG9yeSwgYSBkaXJlY3Rvcnkgd2lsbCBiZSBjcmVhdGVkIGluIHRoZSBmb3JtIGBgYDxkYXRlPl88aWQ+YGBgIHdoaWNoIGNvbnRhaW5zIHRoZSBzaW11bGF0aW9uIHJlc3VsdHMuCjQuIGBgYGNkIDxkYXRlPl88aWQ+YGBgCjUuIGBgYHJlZ3JlcG9ydCAkUFdEID4gcmVwb3J0LmxvZ2BgYCB3aWxsIHByb2Nlc3MgdGhlIHJlc3VsdHMgZnJvbSBlYWNoIG9mIHRoZSByZWdyZXNzaW9ucyBhbmQgcGxhY2UgdGhlIGFnZ3JlZ2F0ZWQgcmVzdWx0cyBpbiB0aGUgZmlsZSBgYGByZXBvcnQubG9nYGBgLiAgRm9yIG1vcmUgZGV0YWlscyBvbiBydW5uaW5nIGEgcmVncmVzc2lvbiwgdGhlIGF2YWlsYWJsZSByZWdyZXNzaW9uIGdyb3VwcywgdW5kZXJzdGFuZGluZyB0aGUgcmVncmVzc2lvbiBvdXRwdXQsIGFuZCBzcGVjaWZ5aW5nIGEgbmV3IHJlZ3Jlc3Npb24gZ3JvdXAsIHBsZWFzZSByZWZlciB0byB0aGUgT3BlblBpdG9uIGRvY3VtZW50YXRpb24uCgo9PT09PT09PT09PT09PT09PT09PT09PT09PQoKIyMjIyBSdW5uaW5nIGEgY29udGludW91cyBpbnRlZ3JhdGlvbiBidW5kbGUKQ29udGludW91cyBpbnRlZ3JhdGlvbiBidW5kbGVzIGFyZSBzZXRzIG9mIHNpbXVsYXRpb25zLCByZWdyZXNzaW9uIGdyb3VwcywgYW5kL29yIHVuaXQgdGVzdHMuICBUaGUgc2ltdWxhdGlvbnMgd2l0aGluIGEgYnVuZGxlIGFyZSBub3QgcmVxdWlyZWQgdG8gaGF2ZSB0aGUgc2FtZSBzaW11bGF0aW9uIG1vZGVsLiAgVGhlIGNvbnRpbnVvdXMgaW50ZWdyYXRpb24gdG9vbCByZXF1aXJlcyBhIGpvYiBxdWV1ZSBtYW5hZ2VyIChlLmcuIFNMVVJNLCBQQlMsIGV0Yy4pIHRvIGJlIHByZXNlbnQgb24gdGhlIHN5c3RlbSBpbiBvcmRlciBwYXJhbGxlbGl6ZSBzaW11bGF0aW9ucy4KCjEuIGBgYGNkICRQSVRPTl9ST09UL2J1aWxkYGBgCjIuIGBgYGNvbnRpbnQgLS1idW5kbGU9Z2l0X3B1c2hgYGAgcnVucyB0aGUgZ2l0X3B1c2ggY29udGludW91cyBpbnRlZ3JhdGlvbiBidW5kbGUgd2hpY2ggd2UgcmFuIG9uIGV2ZXJ5IGNvbW1pdCB3aGVuIGRldmVsb3BpbmcgUGl0b24uICBJdCBjb250YWlucyBhIHJlZ3Jlc3Npb24gZ3JvdXAsIHNvbWUgYXNzZW1ibHkgdGVzdHMsIGFuZCBzb21lIHVuaXQgdGVzdHMuCjMuIFRoZSBzaW11bGF0aW9uIG1vZGVscyB3aWxsIGJlIGJ1aWx0IGFuZCBhbGwgc2ltdWxhdGlvbiBqb2JzIHdpbGwgYmUgc3VibWl0dGVkCjQuIEFmdGVyIGFsbCBzaW11bGF0aW9uIGpvYnMgY29tcGxldGUsIHRoZSByZXN1bHRzIHdpbGwgYmUgYWdncmVnYXRlZCBhbmQgcHJpbnRlZCB0byB0aGUgc2NyZWVuLiAgVGhlIGluZGl2aWR1YWwgc2ltdWxhdGlvbiByZXN1bHRzIHdpbGwgYmUgc2F2ZWQgaW4gYSBuZXcgZGlyZWN0b3J5IGluIHRoZSBmb3JtIGBgYGNvbnRpbnRfPGJ1bmRsZSBuYW1lPl88ZGF0ZT5fPGlkPmBgYCBhbmQgY2FuIGJlIHJlcHJvY2Vzc2VkIGxhdGVyIHRvIHZpZXcgdGhlIGFnZ3JlZ2F0ZWQgcmVzdWx0cyBhZ2Fpbi4KNS4gVGhlIGV4aXQgY29kZSBvZiB0aGUgY29tbWFuZCBpbiBTdGVwIDIgaW5kaWNhdGVzIHdoZXRoZXIgYWxsIHRlc3RzIHBhc3NlZCAoemVybyBleGl0IGNvZGUpIG9yIGF0IGxlYXN0IG9uZSBmYWlsZWQgKG5vbi16ZXJvIGV4aXQgY29kZSkuCjYuIEZvciBtb3JlIGRldGFpbHMgb24gcnVubmluZyBjb250aW51b3VzIGludGVncmF0aW9uIGJ1bmRsZXMsIHRoZSBhdmFpbGFibGUgYnVuZGxlcywgdW5kZXJzdGFuZGluZyB0aGUgb3V0cHV0LCByZXByb2Nlc3NpbmcgY29tcGxldGVkIGJ1bmRsZXMsIGFuZCBjcmVhdGluZyBuZXcgYnVuZGxlcywgcGxlYXNlIHJlZmVyIHRvIHRoZSBPcGVuUGl0b24gZG9jdW1lbnRhdGlvbi4KCj09PT09PT09PT09PT09PT09PT09PT09PT09CiFbT3BlblBpdG9uQXJpYW5lIExvZ29dKC9kb2NzL29wZW5waXRvbl9hcmlhbmVfbG9nby5wbmc/cmF3PXRydWUpCgojIyMjIFByZWxpbWluYXJ5IFN1cHBvcnQgZm9yIEFyaWFuZSBSVjY0SU1BQyBDb3JlCgpUaGlzIHZlcnNpb24gb2YgT3BlblBpdG9uIGhhcyBwcmVsaW1pbmFyeSBzdXBwb3J0IGZvciB0aGUgWzY0Yml0IEFyaWFuZSBSSVNDLVYgcHJvY2Vzc29yXShodHRwczovL2dpdGh1Yi5jb20vcHVscC1wbGF0Zm9ybS9hcmlhbmUpIGZyb20gRVRIIFp1cmljaC4KVG8gdGhpcyBlbmQsIEFyaWFuZSBoYXMgYmVlbiBlcXVpcHBlZCB3aXRoIGEgZGlmZmVyZW50IEwxIGNhY2hlIHN1YnN5c3RlbSB0aGF0IGZvbGxvd3MgYSB3cml0ZS10aHJvdWdoIHByb3RvY29sIGFuZCB0aGF0IGhhcyBzdXBwb3J0IGZvciBjYWNoZSBpbnZhbGlkYXRpb25zIGFuZCBhdG9taWNzLgpUaGlzIEwxIGNhY2hlIHN5c3RlbSBpcyBkZXNpZ25lZCB0byBjb25uZWN0IGRpcmVjdGx5IHRvIHRoZSBMMS41IGNhY2hlIHByb3ZpZGVkIGJ5IE9wZW5QaXRvbidzIFAtTWVzaC4KCkNoZWNrIG91dCB0aGUgc2VjdGlvbnMgYmVsb3cgdG8gc2VlIGhvdyB0byBydW4gdGhlIFJJU0MtViB0ZXN0cyBvciBzaW1wbGUgYmFyZS1tZXRhbCBDIHByb2dyYW1zIGluIHNpbXVsYXRpb24uCgo+IE5vdGUgdGhhdCB0aGUgc3lzdGVtIGhhcyBvbmx5IGJlZW4gdGVzdGVkIHdpdGggYSAxeDEgdGlsZSBjb25maWd1cmF0aW9uLiBWZXJpZmljYXRpb24gb2YgbW9yZSBhZHZhbmNlZCBmZWF0dXJlcyBzdWNoIGFzIGNhY2hlIGNvaGVyZW5jeSBhbW9uZyBtdWx0aXBsZSB0aWxlcyBpcyBzdGlsbCBhIHdvcmstaW4tcHJvZ3Jlc3MsIGFsdGhvdWdoIHNpbXBsZSB0ZXN0IHByb2dyYW1zIGRvIHdvcmsgaW4gdGhlIG1hbnljb3JlIHNldHRpbmcgKHNlZSBiZWxvdykuCgo+IEZvciBzaW11bGF0aW9uLCBRdWVzdGFzaW0gMTAuNmIgaXMgbmVlZGVkIChvbGRlciB2ZXJzaW9ucyBtaWdodCB3b3JrLCBidXQgaGF2ZSBub3QgYmVlbiB0ZXN0ZWQpLgoKPiBZb3Ugd2lsbCBuZWVkIFZpdmFkbyAyMDE3LjMgb3IgbmV3ZXIgdG8gYnVpbGQgYW4gRlBHQSBiaXRzdHJlYW0gd2l0aCBBcmlhbmUuCgohW2Jsb2NrZGlhZ10oL2RvY3Mvb3BlbnBpdG9uX2FyaWFuZV9ibG9ja2RpYWcucG5nP3Jhdz10cnVlKQoKIyMjIyMgRW52aXJvbm1lbnQgU2V0dXAKCkluIGFkZGl0aW9uIHRvIHRoZSBPcGVuUGl0b24gc2V0dXAgZGVzY3JpYmVkIGFib3ZlLCB5b3UgaGF2ZSB0byBhZGFwdCB0aGUgcGF0aHMgaW4gdGhlIGBhcmlhbmVfc2V0dXAuc2hgIHNjcmlwdCB0byBtYXRjaCB3aXRoIHlvdXIgaW5zdGFsbGF0aW9uIChub3RlIHRoYXQgb25seSBRdWVzdGFzaW0gaXMgc3VwcG9ydGVkIGF0IHRoZSBtb21lbnQpLiBTb3VyY2UgdGhpcyBzY3JpcHQgZnJvbSB0aGUgT3BlblBpdG9uIHJvb3QgZm9sZGVyIGFuZCBidWlsZCB0aGUgUklTQy1WIHRvb2xzIHdpdGggYGFyaWFuZV9idWlsZF90b29scy5zaGAgaWYgeW91IGFyZSBydW5uaW5nIHRoaXMgZm9yIHRoZSBmaXJzdCB0aW1lOgoxLiBgYGBjZCAkUElUT05fUk9PVC9gYGAKMi4gYGBgc291cmNlIHBpdG9uL2FyaWFuZV9zZXR1cC5zaGBgYAozLiBgYGBwaXRvbi9hcmlhbmVfYnVpbGRfdG9vbHMuc2hgYGAKClN0ZXAgMy4gd2lsbCB0aGVuIGRvd25sb2FkIGFuZCBjb21waWxlIHRoZSBSSVNDLVYgdG9vbGNoYWluIGFuZCBhc3NlbWJseSB0ZXN0cyBmb3IgeW91LgoKPiBOb3RlIHRoYXQgdGhlIGFkZHJlc3MgbWFwIGlzIGRpZmZlcmVudCBmcm9tIHRoZSBzdGFuZGFyZCBPcGVuUGl0b24gY29uZmlndXJhdGlvbi4gRFJBTSBpcyBtYXBwZWQgdG8gYDB4ODAwMF8wMDAwYCwgaGVuY2UgdGhlIGFzc2VtYmx5IHRlc3RzIGFuZCBDIHByb2dyYW1zIGFyZSBsaW5rZWQgd2l0aCB0aGlzIG9mZnNldC4gSGF2ZSBhIGxvb2sgYXQgYHBpdG9uL2Rlc2lnbi94aWxpbngvZ2VuZXN5czIvZGV2aWNlc19hcmlhbmUueG1sYCBmb3IgYSBjb21wbGV0ZSBhZGRyZXNzIG1hcHBpbmcgb3ZlcnZpZXcuCgo+IEFsc28gbm90ZSB0aGF0IHdlIHVzZSBhIHNsaWdodGx5IGFkYXB0ZWQgdmVyc2lvbiBvZiBgc3lzY2FsbHMuY2AuIEluc3RlYWQgb2YgdXNpbmcgdGhlIFJJU0MtViBGRVNWUiwgd2UgdXNlIHRoZSBPcGVuUGl0b24gdGVzdGJlbmNoIG1vbml0b3JzIHRvIG9ic2VydmUgd2hldGhlciBhIHRlc3QgaGFzIHBhc3NlZCBvciBub3QuIEhlbmNlIHdlIGFkZGVkIHRoZSBjb3JyZXNwb25kaW5nIHBhc3MvZmFpbCB0cmFwcyB0byB0aGUgZXhpdCBmdW5jdGlvbiBpbiBgc3lzY2FsbHMuY2AuCgojIyMjIyBSdW5uaW5nIFJJU0MtViBUZXN0cyBhbmQgQmVuY2htYXJrcwoKVGhlIFJJU0MtViBiZW5jaG1hcmtzIGFyZSBwcmVjb21waWxlZCBpbiB0aGUgdG9vbCBzZXR1cCBzdGVwIG1lbnRpb25lZCBhYm92ZS4gWW91IGNhbiBydW4gaW5kaXZpZHVhbCBiZW5jaG1hcmtzIGJ5IGZpcnN0IGJ1aWxkaW5nIHRoZSBzaW11bGF0aW9uIG1vZGVsIHdpdGgKCjEuIGBgYGNkICRQSVRPTl9ST09UL2J1aWxkYGBgCjIuIGBgYHNpbXMgLXN5cz1tYW55Y29yZSAteF90aWxlcz0xIC15X3RpbGVzPTEgLW1zbV9idWlsZCAtYXJpYW5lYGBgCgpUaGVuLCBpbnZva2UgYSBzcGVjaWZpYyByaXNjdiB0ZXN0IHdpdGggdGhlIGAtcHJlY29tcGlsZWRgIHN3aXRjaCBhcyBmb2xsb3dzCgpgYGBzaW1zIC1zeXM9bWFueWNvcmUgLW1zbV9ydW4gLXhfdGlsZXM9MSAteV90aWxlcz0xIHJ2NjR1aS1wLWFkZGkuUyAtYXJpYW5lIC1wcmVjb21waWxlZGBgYAoKVGhpcyB3aWxsIGxvb2sgZm9yIHRoZSBwcmVjb21waWxlZCBJU0EgdGVzdCBiaW5hcnkgbmFtZWQgYHJ2NjR1aS1wLWFkZGlgIGluIHRoZSBSSVNDLVYgdGVzdHMgZm9sZGVyIGAkQVJJQU5FX1JPT1QvdG1wL3Jpc2N2LXRlc3RzL2J1aWxkL2lzYWAgYW5kIHJ1biBpdC4KCkluIG9yZGVyIHRvIHJ1biBhIFJJU0MtViBiZW5jaG1hcmssIGRvCgpgYGBzaW1zIC1zeXM9bWFueWNvcmUgLW1zbV9ydW4gLXhfdGlsZXM9MSAteV90aWxlcz0xIGRocnlzdG9uZS5yaXNjdiAtYXJpYW5lIC1wcmVjb21waWxlZGBgYAoKVGhlIHByaW50ZiBvdXRwdXQgd2lsbCBiZSBkaXJlY3RlZCB0byBgZmFrZV91YXJ0LmxvZ2AgaW4gdGhpcyBjYXNlIChpbiB0aGUgYnVpbGQgZm9sZGVyKS4KCj4gTm90ZTogaWYgeW91IHNlZSB0aGUgYFdhcm5pbmc6IFtsMTVfYWRhcHRlcl0gcmV0dXJuIHR5cGUgMDA0IGlzIG5vdCAoeWV0KSBzdXBwb3J0ZWQgYnkgbDE1IGFkYXB0ZXIuYCB3YXJuaW5nIGluIHRoZSBzaW11bGF0aW9uIG91dHB1dCwgZG8gbm90IHdvcnJ5LiBUaGlzIGlzIG9ubHkgZ2VuZXJhdGVkIHNpbmNlIEFyaWFuZSBkb2VzIGN1cnJlbnRseSBub3Qgc3VwcG9ydCBPcGVuUGl0b24ncyBwYWNrZXQtYmFzZWQgaW50ZXJydXB0IHBhY2tldHMgYXJyaXZpbmcgb3ZlciB0aGUgbWVtb3J5IGludGVyZmFjZS4KCgoKIyMjIyMgUnVubmluZyBDdXN0b20gUHJvZ3JhbXMKCllvdSBjYW4gYWxzbyBydW4gdGVzdCBwcm9ncmFtcyB3cml0dGVuIGluIEMuIFRoZSBmb2xsb3dpbmcgZXhhbXBsZSBwcm9ncmFtIGp1c3QgcHJpbnRzIDMyIHRpbWVzICJoZWxsb193b3JsZCIgdG8gdGhlIGZha2UgVUFSVCAoc2VlIGBmYWtlX3VhcnQubG9nYCBmaWxlKS4KCjEuIGBgYGNkICRQSVRPTl9ST09UL2J1aWxkYGBgCjIuIGBgYHNpbXMgLXN5cz1tYW55Y29yZSAteF90aWxlcz0xIC15X3RpbGVzPTEgLW1zbV9idWlsZCAtYXJpYW5lYGBgCjMuIGBgYHNpbXMgLXN5cz1tYW55Y29yZSAtbXNtX3J1biAteF90aWxlcz0xIC15X3RpbGVzPTEgaGVsbG9fd29ybGQuYyAtYXJpYW5lIC1ydGxfdGltZW91dCAxMDAwMDAwMGBgYAoKQW5kIGEgc2ltcGxlIGhlbGxvIHdvcmxkIHByb2dyYW0gcnVubmluZyBvbiBtdWx0aXBsZSB0aWxlcyBjYW4gcnVuIGFzIGZvbGxvd3M6CgoxLiBgYGBjZCAkUElUT05fUk9PVC9idWlsZGBgYAoyLiBgYGBzaW1zIC1zeXM9bWFueWNvcmUgLXhfdGlsZXM9NCAteV90aWxlcz00IC1tc21fYnVpbGQgLWFyaWFuZWBgYAozLiBgYGBzaW1zIC1zeXM9bWFueWNvcmUgLW1zbV9ydW4gLXhfdGlsZXM9NCAteV90aWxlcz00ICBoZWxsb193b3JsZF9tYW55LmMgLWFyaWFuZSAtZmluaXNoX21hc2sgMHgxMTExMTExMTExMTExMTExIC1ydGxfdGltZW91dCAxMDAwMDAwYGBgCgpJbiB0aGUgZXhhbXBsZSBhYm92ZSwgd2UgaGF2ZSBhIDR4NCBBcmlhbmUgdGlsZSBjb25maWd1cmF0aW9uLCB3aGVyZSBlYWNoIGNvcmUganVzdCBwcmludHMgaXRzIG93biBoYXJ0IElEIChoYXJkd2FyZSB0aHJlYWQgSUQpIHRvIHRoZSBmYWtlIFVBUlQuIFN5bmNocm9uaXphdGlvbiBhbW9uZyB0aGUgaGFydHMgaXMgYWNoaWV2ZWQgdXNpbmcgYW4gYXRvbWljIEFERCBvcGVyYXRpb24uCgo+IE5vdGUgdGhhdCB3ZSBoYXZlIHRvIGFkanVzdCB0aGUgZmluaXNoIG1hc2sgaW4gdGhpcyBjYXNlLCBzaW5jZSB3ZSBleHBlY3QgYWxsIDE2IGNvcmVzIHRvIGhpdCB0aGUgcGFzcy9mYWlsIHRyYXAuCgojIyMjIyBSZWdyZXNzaW9ucwoKVGhlIFJJU0MtViBJU0EgdGVzdHMsIGJlbmNobWFya3MgYW5kIHNvbWUgYWRkaXRvbmFsIHNpbXBsZSBleGFtcGxlIHByb2dyYW1zIGhhdmUgYmVlbiBhZGRlZCB0byB0aGUgcmVncmVzc2lvbiBzdWl0ZSBvZiBPcGVuUGl0b24sIGFuZCBjYW4gYmUgaW52b2tlZCBhcyBkZXNjcmliZWQgYmVsb3cuCgotIFJJU0MtViBJU0EgdGVzdHMgYXJlIGdyb3VwZWQgaW50byB0aGUgZm9sbG93aW5nIGZvdXIgYmF0Y2hlcywgd2hlcmUgdGhlIGxhc3QgdHdvIGFyZSB0aGUgcmVncmVzc2lvbnMgZm9yIGF0b21pYyBtZW1vcnkgb3BlcmF0aW9ucyAoQU1Pcyk6CgpgYGBzaW1zIC1ncm91cD1hcmlhbmVfdGlsZTFfYXNtX3Rlc3RzX3AgLXNpbV90eXBlPW1zbWBgYAoKYGBgc2ltcyAtZ3JvdXA9YXJpYW5lX3RpbGUxX2FzbV90ZXN0c192IC1zaW1fdHlwZT1tc21gYGAKCmBgYHNpbXMgLWdyb3VwPWFyaWFuZV90aWxlMV9hbW9fdGVzdHNfcCAtc2ltX3R5cGU9bXNtYGBgCgpgYGBzaW1zIC1ncm91cD1hcmlhbmVfdGlsZTFfYW1vX3Rlc3RzX3YgLXNpbV90eXBlPW1zbWBgYAoKLSBSSVNDLVYgYmVuY2htYXJrcyBjYW4gYmUgcnVuIHdpdGg6CgpgYGBzaW1zIC1ncm91cD1hcmlhbmVfdGlsZTFfYmVuY2htYXJrcyAtc2ltX3R5cGU9bXNtYGBgCgotIFNpbXBsZSBoZWxsbyB3b3JsZCBwcm9ncmFtcyBhbmQgQU1PIHRlc3RzIGZvciAxIHRpbGUgY2FuIGJlIGludm9rZWQgd2l0aAoKYGBgc2ltcyAtZ3JvdXA9YXJpYW5lX3RpbGUxX3NpbXBsZSAtc2ltX3R5cGU9bXNtYGBgCgotIEFuZCBhIG11bHRpY29yZSAiaGVsbG8gd29ybGQiIGV4YW1wbGUgcnVubmluZyBvbiAxNiB0aWxlcyBjYW4gYmUgcnVuIHdpdGgKCmBgYHNpbXMgLWdyb3VwPWFyaWFuZV90aWxlMTZfc2ltcGxlIC1zaW1fdHlwZT1tc21gYGAKCgpJZiB5b3Ugd291bGQgbGlrZSB0byBnZXQgYW4gb3ZlcnZpZXcgb2YgdGhlIGV4aXQgc3RhdHVzIG9mIGEgcmVncmVzc2lvbiBiYXRjaCwgc3RlcCBpbnRvIHRoZSByZWdyZXNzaW9uIHN1YmZvbGRlciBhbmQgY2FsbCBgcmVncmVwb3J0IC4gLXN1bW1hcnlgLgoKCiMjIyMjIEZQR0EgTWFwcGluZyBvbiBHZW5lc3lzMiBCb2FyZAoKVGhlIGJpdGZpbGUgZm9yIGEgMXgxIHRpbGUgQXJpYW5lIGNvbmZpZ3VyYXRpb24gZm9yIHRoZSBHZW5lc3lzMiBib2FyZCBjYW4gYmUgYnVpbHQgdXNpbmcgdGhlIGZvbGxvbmcgY29tbWFuZDoKCmBgYHByb3Rvc3luIC1iIGdlbmVzeXMyIC1kIHN5c3RlbSAtLWNvcmU9YXJpYW5lIC0tdWFydC1kbXcgZGRyYGBgCgo+IEl0IGlzIHJlY29tbWVuZGVkIHRvIHVzZSBWaXZhZG8gMjAxOC4yIHNpbmNlIG90aGVyIHZlcnNpb25zIG1pZ2h0IG5vdCBwcm9kdWNlIGEgd29ya2luZyBiaXRzdHJlYW0uCgpPbmNlIHlvdSBoYXZlIGxvYWRlZCB0aGUgYml0c3RyZWFtIG9udG8gdGhlIEZQR0EgdXNpbmcgdGhlIFZpdmFkbyBIYXJkd2FyZSBNYW5hZ2VyIG9yIGEgVVNCIGRyaXZlIHBsdWdnZWQgaW50byB0aGUgR2VuZXN5czIsIHlvdSBmaXJzdCBuZWVkIHRvIGNvbm5lY3QgdGhlIFVBUlQvVVNCIHBvcnQgb2YgdGhlIEdlbmVzeXMyIGJvYXJkIHRvIHlvdXIgY29tcHV0ZXIgYW5kIGZsaXAgc3dpdGNoIDcgb24gdGhlIGJvYXJkIGFzIGRlc2NyaWJlZCBpbiB0aGUgW09wZW5QaXRvbiBGUEdBIFByb3RvdHlwZSBNYW51YWxdKGh0dHA6Ly9wYXJhbGxlbC5wcmluY2V0b24uZWR1L29wZW5waXRvbi9kb2NzL2ZwZ2FfbWFuLnBkZikuIFRoZW4geW91IGNhbiB1c2UgcGl0b25zdHJlYW0gdG8gcnVuIGEgbGlzdCBvZiB0ZXN0cyBvbiB0aGUgRlBHQToKCmBgYHBpdG9uc3RyZWFtIC1iIGdlbmVzeXMyIC1kIHN5c3RlbSAtZiAuL3Rlc3RzLnR4dCAtLWNvcmU9YXJpYW5lYGBgCgpUaGUgdGVzdHMgdGhhdCB5b3Ugd291bGQgbGlrZSB0byBydW4gbmVlZCB0byBiZSBzcGVjaWZpZWQgaW4gdGhlIGB0ZXN0LnR4dGAgZmlsZSwgb25lIHRlc3QgcGVyIGxpbmUgKGUuZy4gYGhlbGxvX3dvcmxkLmNgKS4KCllvdSBjYW4gYWxzbyBydW4gdGhlIHByZWNvbXBpbGVkIFJJU0NWIGJlbmNobWFya3MgYnkgdXNpbmcgdGhlIGZvbGxvd2luZyBjb21tYW5kCgpgYGBwaXRvbnN0cmVhbSAtYiBnZW5lc3lzMiAtZCBzeXN0ZW0gLWYgLi9waXRvbi9kZXNpZ24vY2hpcC90aWxlL2FyaWFuZS9jaS9yaXNjdi1iZW5jaG1hcmtzLmxpc3QgLS1jb3JlPWFyaWFuZSAtLXByZWNvbXBpbGVkYGBgCgo+IE5vdGUgdGhlIGAtcHJlY29tcGlsZWRgIHN3aXRjaCBoZXJlLCB3aGljaCBoYXMgdGhlIHNhbWUgZWZmZWN0IGFzIHdoZW4gdXNlZCB3aXRoIHRoZSBgc2ltc2AgY29tbWFuZC4KCiMjIyMjIERlYnVnZ2luZyB2aWEgSlRBRwoKT3BlblBpdG9uK0FyaWFuZSBzdXBwb3J0cyB0aGUgW1JJU0MtViBFeHRlcm5hbCBEZWJ1ZyBEcmFmdCBTcGVjXShodHRwczovL2dpdGh1Yi5jb20vcmlzY3YvcmlzY3YtZGVidWctc3BlYy9ibG9iL21hc3Rlci9yaXNjdi1kZWJ1Zy1kcmFmdC5wZGYpIGFuZCBoZW5jZSB5b3UgY2FuIGRlYnVnIChhbmQgcHJvZ3JhbSkgdGhlIEZQR0EgdXNpbmcgW09wZW5PQ0RdKGh0dHA6Ly9vcGVub2NkLm9yZy9kb2MvaHRtbC9BcmNoaXRlY3R1cmUtYW5kLUNvcmUtQ29tbWFuZHMuaHRtbCkuIFdlIHByb3ZpZGUgdHdvIGV4YW1wbGUgc2NyaXB0cyBmb3IgT3Blbk9DRCBiZWxvdy4KClRvIGdldCBzdGFydGVkLCBjb25uZWN0IHRoZSBtaWNybyBVU0IgcG9ydCB0aGF0IGlzIGxhYmVsZWQgd2l0aCBKVEFHIHRvIHlvdXIgbWFjaGluZS4gVGhpcyBwb3J0IGlzIGF0dGFjaGVkIHRvIHRoZSBGVERJIDIyMzIgVVNCLXRvLXNlcmlhbCBjaGlwIG9uIHRoZSBHZW5lc3lzIDIgYm9hcmQsIGFuZCBpcyB1c3VhbGx5IHVzZWQgdG8gYWNjZXNzIHRoZSBuYXRpdmUgSlRBRyBpbnRlcmZhY2Ugb2YgdGhlIEtpbnRleC03IEZQR0EgKGUuZy4gdG8gcHJvZ3JhbSB0aGUgZGV2aWNlIHVzaW5nIFZpdmFkbykuIEhvd2V2ZXIsIHRoZSBGVERJIGNoaXAgYWxzbyBleHBvc2VzIGEgc2Vjb25kIHNlcmlhbCBsaW5rIHRoYXQgaXMgcm91dGVkIHRvIEdQSU8gcGlucyBvbiB0aGUgRlBHQSwgYW5kIHdlIGxldmVyYWdlIHRoaXMgdG8gd2lyZSB1cCB0aGUgSlRBRyBmcm9tIHRoZSBSSVNDLVYgZGVidWcgbW9kdWxlLgoKPklmIHlvdSBhcmUgb24gYW4gVWJ1bnR1IGJhc2VkIHN5c3RlbSB5b3UgbmVlZCB0byBhZGQgdGhlIGZvbGxvd2luZyB1ZGV2IHJ1bGUgdG8gYC9ldGMvdWRldi9ydWxlcy5kLzk5LWZ0ZGkucnVsZXNgCj5gYGAKPiBTVUJTWVNURU09PSJ1c2IiLCBBQ1RJT049PSJhZGQiLCBBVFRSU3tpZFByb2R1Y3R9PT0iNjAxMCIsIEFUVFJTe2lkVmVuZG9yfT09IjA0MDMiLCBNT0RFPSI2NjQiLCBHUk9VUD0icGx1Z2RldiIKPmBgYAoKT25jZSBhdHRhY2hlZCB0byB5b3VyIHN5c3RlbSwgdGhlIEZUREkgY2hpcCBzaG91bGQgYmUgbGlzdGVkIHdoZW4geW91IHR5cGUgYGxzdXNiYApgYGAKQnVzIDAwNSBEZXZpY2UgMDE5OiBJRCAwNDAzOjYwMTAgRnV0dXJlIFRlY2hub2xvZ3kgRGV2aWNlcyBJbnRlcm5hdGlvbmFsLCBMdGQgRlQyMjMyQy9EL0ggRHVhbCBVQVJUL0ZJRk8gSUMKYGBgCgpJZiB0aGlzIGlzIHRoZSBjYXNlLCB5b3UgY2FuIGdvIG9uIGFuZCBzdGFydCBvcGVub2NkIHdpdGggdGhlIGBmcGdhL2FyaWFuZS5jZmdgIGNvbmZpZ3VyYXRpb24gZmlsZSBiZWxvdy4KYGBgCiQgb3Blbm9jZCAtZiBmcGdhL2FyaWFuZS5jZmcKT3BlbiBPbi1DaGlwIERlYnVnZ2VyIDAuMTAuMCtkZXYtMDAxOTUtZzkzM2NiODcgKDIwMTgtMDktMTQtMTk6MzIpCkxpY2Vuc2VkIHVuZGVyIEdOVSBHUEwgdjIKRm9yIGJ1ZyByZXBvcnRzLCByZWFkCiAgICBodHRwOi8vb3Blbm9jZC5vcmcvZG9jL2RveHlnZW4vYnVncy5odG1sCmFkYXB0ZXIgc3BlZWQ6IDEwMDAga0h6CkluZm8gOiBhdXRvLXNlbGVjdGluZyBmaXJzdCBhdmFpbGFibGUgc2Vzc2lvbiB0cmFuc3BvcnQgImp0YWciLiBUbyBvdmVycmlkZSB1c2UgJ3RyYW5zcG9ydCBzZWxlY3QgPHRyYW5zcG9ydD4nLgpJbmZvIDogY2xvY2sgc3BlZWQgMTAwMCBrSHoKSW5mbyA6IFRBUCByaXNjdi5jcHUgZG9lcyBub3QgaGF2ZSBJRENPREUKSW5mbyA6IGRhdGFjb3VudD0yIHByb2didWZzaXplPTgKSW5mbyA6IEV4YW1pbmVkIFJJU0MtViBjb3JlOyBmb3VuZCAxIGhhcnRzCkluZm8gOiAgaGFydCAwOiBYTEVOPTY0LCBtaXNhPTB4ODAwMDAwMDAwMDE0MTEwNQpJbmZvIDogTGlzdGVuaW5nIG9uIHBvcnQgMzMzMyBmb3IgZ2RiIGNvbm5lY3Rpb25zClJlYWR5IGZvciBSZW1vdGUgQ29ubmVjdGlvbnMKSW5mbyA6IExpc3RlbmluZyBvbiBwb3J0IDY2NjYgZm9yIHRjbCBjb25uZWN0aW9ucwpJbmZvIDogTGlzdGVuaW5nIG9uIHBvcnQgNDQ0NCBmb3IgdGVsbmV0IGNvbm5lY3Rpb25zCkluZm8gOiBhY2NlcHRpbmcgJ2dkYicgY29ubmVjdGlvbiBvbiB0Y3AvMzMzMwpgYGAKTm90ZSB0aGF0IHRoaXMgc2ltcGxlIE9wZW5PQ0Qgc2NyaXB0IGN1cnJlbnRseSBvbmx5IHN1cHBvcnRzIG9uZSBoYXJ0IHRvIGJlIGRlYnVnZ2VkIGF0IGEgdGltZS4gU2VsZWN0IHRoZSBoYXJ0IHRvIGRlYnVnIGJ5IGNoYW5naW5nIHRoZSBjb3JlIGlkIChsb29rIGZvciB0aGUgYC1jb3JlaWRgIHN3aXRjaCBpbiB0aGUgYGFyaWFuZS5jZmdgIGZpbGUpLiAKClRoZW4geW91IHdpbGwgYmUgYWJsZSB0byBlaXRoZXIgY29ubmVjdCB0aHJvdWdoIGB0ZWxuZXRgIG9yIHdpdGggYGdkYmA6CmBgYAokIHJpc2N2NjQtdW5rbm93bi1lbGYtZ2RiIC9wYXRoL3RvL2VsZgooZ2RiKSB0YXJnZXQgcmVtb3RlIGxvY2FsaG9zdDozMzMzCihnZGIpIGxvYWQKTG9hZGluZyBzZWN0aW9uIC50ZXh0LCBzaXplIDB4NjUwOCBsbWEgMHg4MDAwMDAwMApMb2FkaW5nIHNlY3Rpb24gLnJvZGF0YSwgc2l6ZSAweDkwMCBsbWEgMHg4MDAwNjUwOAooZ2RiKSBiIHB1dGNoYXIKKGdkYikgYwpDb250aW51aW5nLgoKUHJvZ3JhbSByZWNlaXZlZCBzaWduYWwgU0lHVFJBUCwgVHJhY2UvYnJlYWtwb2ludCB0cmFwLgoweDAwMDAwMDAwODAwMDkxMjYgaW4gcHV0Y2hhciAocz03MikgYXQgbGliL3FwcmludGYuYzo2OQo2OSAgICB1YXJ0X3NlbmRjaGFyKHMpOwooZ2RiKSBzaQoweDAwMDAwMDAwODAwMDkxMmEgIDY5ICAgIHVhcnRfc2VuZGNoYXIocyk7CihnZGIpIHAveCAkbWVwYwokMSA9IDB4ZmZmZmZmZmZmZmZkYjVlZQpgYGAKCllvdSBjYW4gcmVhZCBvciB3cml0ZSBkZXZpY2UgbWVtb3J5IGJ5IHVzaW5nOgpgYGAKKGdkYikgeC9pIDB4MTAwMAogICAgMHgxMDAwOiBsdWkgdDAsMHg0CihnZGIpIHNldCB7aW50fSAweDEwMDAgPSAyMgooZ2RiKSBzZXQgJHBjID0gMHgxMDAwCmBgYAoKSW4gb3JkZXIgdG8gY29tcGlsZSBwcm9ncmFtcyB0aGF0IHlvdSBjYW4gbG9hZCB3aXRoIEdEQiwgdXNlIHRoZSBmb2xsb3dpbmcgY29tbWFuZDoKCmBgYHNpbXMgLXN5cz1tYW55Y29yZSAtbm92Y3NfYnVpbGQgLW1pZGFzX29ubHkgaGVsbG9fd29ybGQuYyAtYXJpYW5lIC14X3RpbGVzPTEgLXlfdGlsZXM9MSAtZ2NjX2FyZ3M9Ii1nImBgYAoKTm90ZSB0aGF0IHRoZSB0aWxlIGNvbmZpZ3VyYXRpb24gbmVlZHMgdG8gY29ycmVzcG9uZCB0byB5b3VyIGFjdHVhbCBwbGF0Zm9ybSBjb25maWd1cmF0aW9uIGlmIHlvdXIgcHJvZ3JhbSBpcyBhIG11bHRpLWhhcnQgcHJvZ3JhbS4gT3RoZXJ3aXNlIHlvdSBjYW4gb21pdCB0aGVzZSBzd2l0Y2hlcyAodGhlIGFkZGl0aW9uYWwgY29yZXMgd2lsbCBub3QgZXhlY3V0ZSB0aGUgcHJvZ3JhbSBpbiB0aGF0IGNhc2UpLgoKIyMjIyMgQm9vdGluZyBTTVAgTGludXggb24gR2VuZXN5czIgYW5kIFZDNzA3CgpXZSBjdXJyZW50bHkgc3VwcG9ydCBzaW5nbGUgY29yZSBhbmQgU01QIExpbnV4IG9uIHRoZSBHZW5lc3lzMiwgVkM3MDcgYW5kIFZDVTExOCBGUEdBIGRldmVsb3BtZW50IGJvYXJkcy4gVGhlIHNpbmdsZS1jb3JlIGNvbmZpZ3VyYXRpb25zIGFyZSByZWxhdGl2ZWx5IHN0YWJsZSwgaG93ZXZlciB0aGUgU01QIHZlcnNpb25zIGNhbiBzb21ldGltZXMgY3Jhc2ggZHVyaW5nIGJvb3QuIFRoaXMgaXMgYSBrbm93biBpc3N1ZSBhbmQgd2lsbCBiZSBhZGRyZXNzZWQgaW4gYSBmdXR1cmUgcmVsZWFzZS4KCkluIG9yZGVyIHRvIGJ1aWxkIGFuIEZQR0EgaW1hZ2UgZm9yIHRoZXNlIGJvYXJkcywgdXNlIGVpdGhlciBvZiB0aGUgZm9sbG93aW5nIGNvbW1hbmRzOgoKYGBgcHJvdG9zeW4gLWIgZ2VuZXN5czIgLWQgc3lzdGVtIC0tY29yZT1hcmlhbmUgLS11YXJ0LWRtdyBkZHJgYGAKCmBgYHByb3Rvc3luIC1iIHZjNzA3IC1kIHN5c3RlbSAtLWNvcmU9YXJpYW5lIC0tdWFydC1kbXcgZGRyYGBgCgpUaGUgZGVmYXVsdCBwYXJhbWV0ZXJzIGFyZSAxIGNvcmUgZm9yIGFsbCBib2FyZHMsIGJ1dCB5b3UgY2FuIG92ZXJyaWRlIHRoaXMgd2l0aCBjb21tYW5kIGxpbmUgYXJndW1lbnRzLiBUaGUgY29tbWFuZHMgYmVsb3cgcmVwcmVzZW50IHRoZSBtYXhpbXVtIGNvbmZpZ3VyYXRpb25zIHRoYXQgY2FuIGJlIG1hcHBlZCBvbnRvIHRoZSBjb3JyZXNwb25kaW5nIGJvYXJkOgoKYGBgcHJvdG9zeW4gLWIgZ2VuZXN5czIgLWQgc3lzdGVtIC0tY29yZT1hcmlhbmUgLS11YXJ0LWRtdyBkZHIgLS14X3RpbGVzPTJgYGAKCmBgYHByb3Rvc3luIC1iIHZjNzA3IC1kIHN5c3RlbSAtLWNvcmU9YXJpYW5lIC0tdWFydC1kbXcgZGRyIC0teF90aWxlcz0yIC0teV90aWxlcz0yYGBgCgpPbmNlIHlvdSBnZW5lcmF0ZWQgdGhlIEZQR0EgYml0ZmlsZSwgZ28gYW5kIGdyYWIgdGhlIFthcmlhbmUtc2RrXShodHRwczovL2dpdGh1Yi5jb20vcHVscC1wbGF0Zm9ybS9hcmlhbmUtc2RrKSBhbmQgZm9sbG93IHRoZSBzdGVwcyBpbiB0aGF0IHJlYWRtZSB0byBidWlsZCB0aGUgTGludXggaW1hZ2UgYW5kIHByZXBhcmUgdGhlIFNEIGNhcmQgKG1ha2Ugc3VyZSB5b3UgdXNlIHRoZSBgb3BlbnBpdG9uYCBicmFuY2ggaW4gdGhhdCByZXBvc2l0b3J5KS4gSWYgeW91IGRvIG5vdCB3YW50IHRvIGdvIHRocm91Z2ggdGhlIGhhc3NsZSBvZiBidWlsZGluZyB5b3VyIG93biBpbWFnZSwgeW91IGNhbiBkb3dubG9hZCBhIHByZS1idWlsdCBsaW51eCBpbWFnZSBmcm9tIFtoZXJlXShodHRwczovL2dpdGh1Yi5jb20vcHVscC1wbGF0Zm9ybS9hcmlhbmUtc2RrL3JlbGVhc2VzL3RhZy92MC4yLjAtb3ApLgoKPiBOb3RlIHRoYXQgdGhlIGJvYXJkIHNwZWNpZmljIHNldHRpbmdzIGFyZSBlbmNvZGVkIGluIHRoZSBkZXZpY2UgdHJlZSB0aGF0IGlzIGF1dG9tYXRpY2FsbHkgZ2VuZXJhdGVkIGFuZCBjb21waWxlZCBpbnRvIHRoZSBGUEdBIGJpdGZpbGUsIHNvIG5vIHNwZWNpZmljIGNvbmZpZ3VyYXRpb24gb2YgdGhlIExpbnV4IGtlcm5lbCBpcyBuZWVkZWQuCgpJbnNlcnQgdGhlIFNEIGNhcmQgaW50byB0aGUgY29ycmVzcG9uZGluZyBzbG90IG9mIHRoZSBGUEdBIGJvYXJkLCBjb25uZWN0IGEgdGVybWluYWwgdG8gdGhlIFVBUlQgdXNpbmcgZS5nLiBgc2NyZWVuIC9kZXYvdHR5VVNCMCAxMTUyMDBgLCBhbmQgcHJvZ3JhbSB0aGUgRlBHQS4gT25jZSB0aGUgZGV2aWNlIGNvbWVzIG91dCBvZiByZXNldCwgdGhlIHplcm8tc3RhZ2UgYm9vdGxvYWRlciBjb3BpZXMgdGhlIExpbnV4IGltYWdlIChpbmNsdWRpbmcgdGhlIGZpcnN0IHN0YWdlIGJvb3Rsb2FkZXIpIGludG8gRFJBTSwgYW5kIGV4ZWN1dGVzIGl0LiBCZSBwYXRpZW50LCBjb3B5aW5nIGZyb20gU0QgdGFrZXMgYSBjb3VwbGUgb2Ygc2Vjb25kcy4KCgo8IS0tICMjIyMjIEJvb3RpbmcgTGludXggb24gR2VuZXN5czIsIFZDNzA3IGFuZCBWQ1UxMTgKCmBgYHByb3Rvc3luIC1iIHZjdTExOCAtZCBzeXN0ZW0gLS1jb3JlPWFyaWFuZSAtLXVhcnQtZG13IGRkcmBgYAoKYGBgcHJvdG9zeW4gLWIgdmN1MTE4IC1kIHN5c3RlbSAtLWNvcmU9YXJpYW5lIC0tdWFydC1kbXcgZGRyIC0teF90aWxlcz00IC0teF90aWxlcz00YGBgCgo+IEZvciB0aGUgVkNVMTE4IGJvYXJkIHlvdSBuZWVkIHRoZSBbUE1PRCBTRCBhZGFwdGVyXShodHRwczovL3N0b3JlLmRpZ2lsZW50aW5jLmNvbS9wbW9kLXNkLWZ1bGwtc2l6ZWQtc2QtY2FyZC1zbG90LykgZnJvbSBEaWdpbGVudCB0byBiZSBhYmxlIHRvIHVzZSBhbiBTRCBjYXJkICh0aGUgc2xvdCBvbiB0aGUgVkNVMTE4IGJvYXJkIGlzIG5vdCBkaXJlY3RseSBjb25uZWN0ZWQgdG8gdGhlIEZQR0EpLiBBcyB0aGUgUE1PRDAgcG9ydCBoYXMgb3Blbi1kcmFpbiBsZXZlbC1zaGlmdGVycywgeW91IGFsc28gaGF2ZSB0byByZXBsYWNlIHRoZSBSMS1SNCBhbmQgUjctOCByZXNpc3RvcnMgd2l0aCA0NzAgT2htIDAyMDEgU01EIHJlc2lzdG9ycyBvbiB0aGUgRGlnaWxlbnQgUE1PRCBTRCBhZGFwdGVyIHRvIG1ha2Ugc3VyZSB0aGF0IHNpZ25hbCByaXNlIHRpbWVzIGFyZSBzaG9ydCBlbm91Z2guIAogLS0+CgojIyMjIyBQbGFubmVkIEltcHJvdmVtZW50cwoKVGhlIGZvbGxvd2luZyBpdGVtcyBhcmUgY3VycmVudGx5IHVuZGVyIGRldmVsb3BtZW50IGFuZCB3aWxsIGJlIHJlbGVhc2VkIHNvb24uCgotIEZsb2F0aW5nIHBvaW50IHN1cHBvcnQuCgotIFRob3JvdWdoIHZhbGlkYXRpb24gb2YgY2FjaGUgY29oZXJlbmNlLgoKLSBSSVNDLVYgRkVTVlIgc3VwcG9ydCBpbiBzaW11bGF0aW9uLgoKLSBTdXBwb3J0IGZvciBzaW11bGF0aW9uIHdpdGggU3lub3BzeXMgVkNTLgoKLSBTeW50aGVzaXMgZmxvdyBmb3IgbGFyZ2UgRlBHQXMuCgotIFBlcmZvcm1hbmNlIGVuaGFuY2VtZW50cyAoY2FjaGUgcmUtcGFyYW1ldGVyaXphdGlvbiwgd3JpdGUtYnVmZmVyIHRocm91Z2hwdXQpLgoKU3RheSB0dW5lZCEK"

def render(text):
    decoded_text = base64.b64decode(text.encode()).decode()
    return markdown.markdown(decoded_text)

def handler(event, context=None):
    html = render(base64_text)
    # print(html)

    return {
        "result": "Render base64 to markdown finished!"
    }


if __name__ == "__main__":
    event = {}
    print(handler(event))