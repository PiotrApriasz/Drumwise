using Drumwise.Application.Common.Models;

namespace Drumwise.Application.Entities;

public class Lesson : BaseAuditableEntity
{
    public string? LessonSubject { get; set; }
    public string? Exercise { get; set; }
}