using Drumwise.Application.Common.Models;

namespace Drumwise.Application.Entities;

public class Homework : BaseAuditableEntity
{
    public string? AssignedTo { get; set; }
    public string? HomeworkTitle { get; set; }
    public string? Exercise { get; set; }
    public DateTime? Deadline { get; set; }
}