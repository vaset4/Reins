from frontends.observe.panels.approval_audit import ApprovalAuditPanel
from frontends.observe.panels.cache_usage import CacheUsagePanel
from frontends.observe.panels.compression_lane import CompressionLanePanel
from frontends.observe.panels.context_inspector import ContextInspectorPanel
from frontends.observe.panels.demo_story import DemoStoryPanel
from frontends.observe.panels.error_cards import ErrorCardsPanel
from frontends.observe.panels.memory_cards import MemoryCardsPanel
from frontends.observe.panels.overview import OverviewPanel
from frontends.observe.panels.prompt_composition import PromptCompositionPanel
from frontends.observe.panels.response_anatomy import ResponseAnatomyPanel
from frontends.observe.panels.skill_activation import SkillActivationPanel
from frontends.observe.panels.state_heatmap import StateHeatmapPanel
from frontends.observe.panels.timeline import TimelinePanel
from frontends.observe.panels.token_accountant import TokenAccountantPanel
from frontends.observe.panels.tool_strip import ToolStripPanel

ALL_PANELS = [
    OverviewPanel(),
    TimelinePanel(),
    ContextInspectorPanel(),
    PromptCompositionPanel(),
    ResponseAnatomyPanel(),
    TokenAccountantPanel(),
    CacheUsagePanel(),
    ToolStripPanel(),
    ApprovalAuditPanel(),
    MemoryCardsPanel(),
    SkillActivationPanel(),
    StateHeatmapPanel(),
    CompressionLanePanel(),
    ErrorCardsPanel(),
    DemoStoryPanel(),
]

__all__ = ["ALL_PANELS"]
