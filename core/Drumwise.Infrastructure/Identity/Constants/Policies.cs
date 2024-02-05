namespace Drumwise.Infrastructure.Identity.Constants;

public abstract class Policies
{
    public const string RequireAdministratorRole = nameof(RequireAdministratorRole);
    public const string RequireTeacherRole = nameof(RequireTeacherRole);
    public const string CanAddHomework = nameof(CanAddHomework);
}