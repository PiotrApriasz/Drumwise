using Drumwise.Application.Common.Models;

namespace Drumwise.Application.Entities;

public class DrumsAudioToEvaluate : BaseAuditableEntity
{
    public bool Uploaded { get; set; }
    public string? GoogleDriveFileId { get; set; }
    public string? TimingEvaluationResult { get; set; }
    public string? DynamicsEvaluationResult { get; set; }
    public bool Evaluated { get; set; }
}