using Drumwise.Domain.Common;

namespace Drumwise.Domain.Entities;

public class Lesson : BaseAuditableEntity
{
    public string? LessonSubject { get; set; }
    public string? Exercise { get; set; }
}