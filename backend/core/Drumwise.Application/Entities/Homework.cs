using Drumwise.Application.Common.Models;

namespace Drumwise.Application.Entities;

public class Homework : BaseAuditableEntity
{
    public required string? AssignedTo { get; set; }
    public required string HomeworkTitle { get; set; }
    public required string Exercise { get; set; }
    public DateTime Deadline { get; set; }
}