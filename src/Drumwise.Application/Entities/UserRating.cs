using Drumwise.Application.Common.Models;

namespace Drumwise.Application.Entities;

public class UserRating : BaseAuditableEntity
{
    public string? AssignedTo { get; set; }
    public int Mark { get; set; }
    public string? Comment { get; set; }
}