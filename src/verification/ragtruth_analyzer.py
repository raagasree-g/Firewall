"""Read-only RAGTruth response/span analytics."""
import json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_ragtruth(path):
    labels=Counter(); types=Counter(); span_counts=[]; lengths=[]; total=0
    for line in Path(path).open(encoding='utf-8'):
        if not line.strip(): continue
        r=json.loads(line); total+=1; labels[r['label']]+=1; spans=r.get('metadata',{}).get('hallucination_spans') or []; span_counts.append(len(spans))
        for span in spans:
            if isinstance(span,dict):
                types[str(span.get('type','UNKNOWN'))]+=1
                start,end=span.get('start'),span.get('end')
                if isinstance(start,int) and isinstance(end,int): lengths.append(max(0,end-start))
    return {"response_count":total,"label_distribution":dict(labels),"hallucinated_response_rate":labels['HALLUCINATED']/total if total else 0,"hallucination_span_count":sum(span_counts),"mean_spans_per_response":sum(span_counts)/total if total else 0,"mean_span_length":sum(lengths)/len(lengths) if lengths else None,"hallucination_type_distribution":dict(types),"observation":"Response-level labels and span annotations remain separate."}

def save_ragtruth_analysis(path, metrics_path, plot_path):
    summary=analyze_ragtruth(path); Path(metrics_path).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plt.bar(summary['label_distribution'].keys(),summary['label_distribution'].values()); plt.title('RAGTruth response labels'); plt.ylabel('Responses'); plt.tight_layout(); plt.savefig(plot_path,dpi=150); plt.close()
    return summary
