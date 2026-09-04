# V9 technical amendment v1.1

The first paid command stopped during M0 validation, before model loading and
before any v9 forward. Three text hashes had been calculated from a Windows
working tree with CRLF conversion. The Linux Pod contains the canonical LF Git
blob bytes, so those hashes differed even though Git content was identical.

V1.1 changes validation to SHA-256 of `git show HEAD:<path>` bytes and records
the corresponding canonical hashes. It changes no scientific data, prompt,
cue mask, feature, optimizer, null, threshold, decision rule, or budget. The
already staged exact checkpoint remains on Pod `v6w0o4uykl6csw`; the Pod may be
reused after checking out the v1.1 tag.

